import numpy as np
import pytest

import enoki as ek
import mitsuba


def dict_sensor(direction=None, 
                origin_type=None,
                origin_center=[0,0,0] ,
                origin_radius=0,
                origin_a=[0,0,0],
                origin_b=[0,0,0],
                fwidth=1):
    result = {"type": "distant", "origin": "disk", "origin_radius": 0, "origin_center": [0,0,0]}

    if direction:
        result["direction"] = direction

    if origin_type == "disk":
        result["origin"] = origin_type
        result["origin_center"] = origin_center
        result["origin_radius"] = origin_radius
    elif origin_type =="rectangle":
        result["origin"] = origin_type
        result["origin_a"] = origin_a
        result["origin_b"] = origin_b

    result["film"] = {
        "type": "hdrfilm",
        "width": fwidth,
        "height": 1,
        "rfilter": {"type": "box"}
    }

    return result


def make_sensor(**kwargs):
    from mitsuba.core.xml import load_dict
    sensor_dict = dict_sensor(**kwargs)
    return load_dict(sensor_dict).expand()[0]


def test_construct(variant_scalar_rgb):
    from mitsuba.core.xml import load_string, load_dict

    # Construct without parameters (wrong film size)
    with pytest.raises(RuntimeError):
        sensor = load_dict({"type": "distant"}).expand()[0]

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


@pytest.mark.parametrize("origin", [
    {"origin_type": None},
    {"origin_type": "disk", "origin_radius": 0, "origin_center": [0.0, 0.0, 0.0]},
    {"origin_type": "disk", "origin_radius": 0.5, "origin_center": [4.0, 1.0, 0.0]},
    {"origin_type": "rectangle", "origin_a": [0,0,0], "origin_b": [0,0,0]},
    {"origin_type": "rectangle", "origin_a": [-1,-1,0], "origin_b": [1,1,0]}
])
@pytest.mark.parametrize("direction", [
    [0.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
    [2.0, 0.0, 0.0]
])
@pytest.mark.parametrize("ray_kind", ["regular", "differential"])
def test_sample_ray(variant_scalar_rgb, direction, origin, ray_kind):
    kwargs = origin
    kwargs["direction"] = direction
    sensor = make_sensor(**kwargs)

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


def make_scene(**kwargs):
    from mitsuba.core.xml import load_dict

    dict_scene = {
        "type": "scene",
        "sensor": dict_sensor(**kwargs),
        "surface": {"type": "rectangle"}
    }

    return load_dict(dict_scene)


@pytest.mark.parametrize("origin", [
    {"origin_type": "rectangle", "origin_a": [-1,-1,1], "origin_b": [1,1,1], "expected_invalid":0.},
    {"origin_type": "rectangle", "origin_a": [-2,-2,2], "origin_b": [2,2,2], "expected_invalid":0.75},
    {"origin_type": "disk", "origin_center": [0,0,1], "origin_radius": 1, "expected_invalid": 0},
    {"origin_type": "disk", "origin_center": [0,0,2], "origin_radius": 2, "expected_invalid": 0.6816}
])
def test_origin_area(variant_scalar_rgb, origin):
    """Test if the sensor correctly targets the expected area by computing
    the fraction of rays with valid intersections"""
    direction = [0,0,-1]
    expected_invalid = origin.pop("expected_invalid")
    scene = make_scene(direction=direction, **origin)
    sensor = scene.sensors()[0]
    sampler = sensor.sampler()

    n_rays = 10000
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
    # around the origin specified
    if origin["origin_type"] == "disk":
        center = origin["origin_center"]
    elif origin["origin_type"] == "rectangle":
        center = (np.array(origin["origin_b"]) + np.array(origin["origin_a"])) / 2.

    # Average intersection locations should be (in average) centered
    # around (0, 0, 0)
    isect_valid = isect[~np.isnan(isect).all(axis=1)]
    mean_location = isect_valid.mean(axis=0)
    assert np.allclose(mean_location[0], center[0], atol=5e-2)
    assert np.allclose(mean_location[1], center[1], atol=5e-2)
    assert np.allclose(mean_location[2], 0., atol=5e-2)

    n_invalid = np.count_nonzero(np.isnan(isect).all(axis=1))

    assert np.allclose(n_invalid/n_rays, expected_invalid, atol=1e-2)


@pytest.mark.parametrize("w_e", [[0, 0, -1], [0, 1, -1]])
@pytest.mark.parametrize("w_o", [[0, 0, -1], [0, 1, -1]])
@pytest.mark.parametrize("origin", [
    {},
    {"origin": "disk", "origin_center": [0,0,2], "origin_radius": 1},
    {"origin": "rectangle", "origin_a": [-1,-1,2], "origin_b": [1,1,2]}
])
def test_render(variant_scalar_mono, w_e, w_o, origin):
    # Test render results with a simple scene
    from mitsuba.core.xml import load_dict
    from mitsuba.core import Bitmap, Struct, ScalarTransform4f

    l_e = 1.0  # Emitted radiance
    w_e = list(ek.normalize(w_e))  # Emitter direction
    w_o = list(ek.normalize(w_o))  # Sensor direction
    cos_theta_e = abs(ek.dot(w_e, [0, 0, 1]))
    cos_theta_o = abs(ek.dot(w_o, [0, 0, 1]))

    scale = 1
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

    # set the sensor origin such that rays hit the square
    if not origin:
        pass
    elif origin["origin"] == "disk":
        origin["origin_center"] = [-w_o[i]*2 for i in range(3)]
    elif origin["origin"] == "rectangle":
        origin["origin_a"] = [-1-w_o[0]*2, -1-w_o[1]*2, 0-w_o[2]*2]
        origin["origin_b"] = [1-w_o[0]*2, 1-w_o[1]*2, 0-w_o[2]*2]

    dict_scene["sensor"] = {**dict_scene["sensor"], **origin}
    scene = load_dict(dict_scene)

    if not origin:
        origin_area = 2 * np.pi
    elif origin["origin"] == "disk":
        origin_area = (origin["origin_radius"]**2) * np.pi
    elif origin["origin"] =="rectangle":
        origin_a = origin["origin_a"]
        origin_b = origin["origin_b"]
        origin_area = abs(origin_a[0] - origin_b[0]) *  abs(origin_a[1] - origin_b[1])
    
    if origin_area >= surface_area:
        ratio = surface_area / origin_area
    else:
        ratio = origin_area / surface_area

    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    img = np.array(sensor.film().bitmap()).squeeze()
    assert np.allclose(np.array(img), expected * ratio)


     