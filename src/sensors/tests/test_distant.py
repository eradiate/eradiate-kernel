import numpy as np
import pytest

import enoki as ek
import mitsuba


def dict_sensor(direction=None, target=None, fwidth=1):
    result = {"type": "distant", "target": "circle", "target_radius": 0}

    if direction:
        result["direction"] = direction

    if target:
        result["target_center"] = target
        result["target"] = "circle"
    else:
        result["target_center"] = [0, 0, 0]

    result["film"] = {
        "type": "hdrfilm",
        "width": fwidth,
        "height": 1,
        "rfilter": {"type": "box"}
    }

    return result


def make_sensor(direction=None, target=None, fwidth=1):
    from mitsuba.core.xml import load_dict
    return load_dict(dict_sensor(direction, target, fwidth))


def test_construct(variant_scalar_rgb):
    from mitsuba.core.xml import load_string, load_dict

    # Construct without parameters (wrong film size)
    with pytest.raises(RuntimeError):
        sensor = load_dict({"type": "distant"})

    # Construct with wrong film size
    with pytest.raises(RuntimeError):
        sensor = make_sensor(fwidth=2)

    # Construct with minimal parameters
    sensor = make_sensor()
    assert sensor is not None
    assert not sensor.bbox().valid()  # Degenerate bounding box

    # Construct with direction, check transform setup correctness
    world_reference = [[0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]]
    sensor = make_sensor(direction=[0, 0, -1])
    assert ek.allclose(
        sensor.world_transform().eval(0.).matrix,
        world_reference
    )

    sensor = make_sensor(direction=[0, 0, -2])
    assert ek.allclose(
        sensor.world_transform().eval(0.).matrix,
        world_reference
    )


@pytest.mark.parametrize("target", [
    None,
    [0.0, 0.0, 0.0],
    [4.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.5, -0.3, 0.0],

])
@pytest.mark.parametrize("direction", [
    [0.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
    [2.0, 0.0, 0.0]
])
@pytest.mark.parametrize("ray_kind", ["regular", "differential"])
def test_sample_ray(variant_scalar_rgb, direction, target, ray_kind):
    sensor = make_sensor(direction, target=target)

    for (sample1, sample2) in [[[0.32, 0.87], [0.16, 0.44]],
                               [[0.17, 0.44], [0.22, 0.81]],
                               [[0.12, 0.82], [0.99, 0.42]],
                               [[0.72, 0.40], [0.01, 0.61]]]:

        if ray_kind == "regular":
            ray, _ = sensor.sample_ray(1., 1., sample1, sample2, True)
        elif ray_kind == "differential":
            ray, _ = sensor.sample_ray_differential(
                1., 1., sample1, sample2, True)
            assert not ray.has_differentials

        # Check that ray direction is what is expected
        assert ek.allclose(ray.d, ek.normalize(direction))

        # Check that ray origin is outside of bounding sphere
        # Bounding sphere is centered at world origin and has radius 1 without scene
        assert ek.norm(ray.o) >= 1.


def make_scene(direction=[0, 0, -1], target=None):
    from mitsuba.core.xml import load_dict

    dict_scene = {
        "type": "scene",
        "sensor": dict_sensor(direction, target=target),
        "surface": {"type": "rectangle"}
    }

    return load_dict(dict_scene)


@pytest.mark.parametrize("target", [[0, 0, 0], [0.5, 0, 1]])
def test_target(variant_scalar_rgb, target):
    # Check if the sensor correctly targets the point it is supposed to
    scene = make_scene(direction=[0, 0, -1], target=target)
    sensor = scene.sensors()[0]
    sampler = sensor.sampler()

    ray, _ = sensor.sample_ray(
        sampler.next_1d(),
        sampler.next_1d(),
        sampler.next_2d(),
        sampler.next_2d()
    )
    si = scene.ray_intersect(ray)
    assert si.is_valid()
    assert ek.allclose(si.p, [target[0], target[1], 0.], atol=1e-6)


@pytest.mark.parametrize("direction", [[0, 0, -1], [0.5, 0.5, -1]])
def test_intersection(variant_scalar_rgb, direction):
    # Check if the sensor correctly casts rays spread uniformly in the scene
    direction = list(ek.normalize(direction))
    scene = make_scene(direction=direction)
    sensor = scene.sensors()[0]
    sampler = sensor.sampler()

    n_rays = 1000
    isect = np.empty((n_rays, 3))

    for i in range(n_rays):
        ray, _ = sensor.sample_ray(
            sampler.next_1d(),
            sampler.next_1d(),
            sampler.next_2d(),
            sampler.next_2d()
        )
        si = scene.ray_intersect(ray)

        if not si.is_valid():
            isect[i, :] = np.nan
        else:
            isect[i, :] = si.p[:]

    # Average intersection locations should be (in average) centered
    # around (0, 0, 0)
    isect_valid = isect[~np.isnan(isect).all(axis=1)]
    assert np.allclose(isect_valid[:, :2].mean(axis=0), 0., atol=5e-2)
    assert np.allclose(isect_valid[:, 2], 0., atol=1e-5)

    # Check number of invalid intersections
    # We expect a ratio of invalid interactions equal to the square's area
    # divided by the bounding sphere's cross section, weighted by the surface's
    # slanting factor (cos theta) w.r.t the sensor's direction
    n_invalid = np.count_nonzero(np.isnan(isect).all(axis=1))
    assert np.allclose(n_invalid / n_rays, 1. - 2. / np.pi *
                       ek.dot(direction, [0, 0, -1]), atol=0.1)


def test_render(variant_scalar_rgb):
    # Test render results with a simple scene
    from mitsuba.core.xml import load_dict
    from mitsuba.core import Bitmap, Struct, ScalarTransform4f

    for w_e, w_o in zip(([0, 0, -1], [0, 1, -1]), ([0, 0, -1], [0, 1, -1])):
        l_e = 1.0  # Emitted radiance
        w_e = list(ek.normalize(w_e))  # Emitter direction
        w_o = list(ek.normalize(w_o))  # Sensor direction
        cos_theta_e = abs(ek.dot(w_e, [0, 0, 1]))
        cos_theta_o = abs(ek.dot(w_o, [0, 0, 1]))

        scale = 0.5
        rho = 1.0  # Surface reflectance
        surface_area = 4. * scale ** 2

    expected = l_e * cos_theta_e * surface_area * rho / np.pi * cos_theta_o

    dict_scene = {
        "type": "scene",
        "shape": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(scale),
                "bsdf": {"type": "diffuse", "reflectance": rho},
        },
        "emitter": {
            "type": "directional",
            "irradiance": l_e,
            "direction": w_e
        },
        "sensor": {
            "type": "distant",
            "direction": w_o,
            "film": {
                    "type": "hdrfilm",
                    "height": 1,
                    "width": 1,
                    "pixel_format": "luminance",
                    "rfilter": {"type": "box"},
            },
            "sampler": {
                "type": "independent",
                "sample_count": 512
            },
        },
        "integrator": {"type": "path"}
    }

    scene = load_dict(dict_scene)
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    img = np.array(sensor.film().bitmap()).squeeze()
    assert np.allclose(np.array(img), expected)
