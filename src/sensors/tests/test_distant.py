import numpy as np
import pytest

import enoki as ek
import mitsuba


def xml_sensor(direction=None, target=None, xpixel=1):
    if direction is None:
        xml_direction = ""
    else:
        if type(direction) is not str:
            direction = ",".join([str(x) for x in direction])
        xml_direction = f"""<vector name="direction" value="{direction}"/>"""

    if target is None:
        xml_target = ""
    else:
        if type(target) is not str:
            target = ",".join([str(x) for x in target])
        xml_target = f"""<point name="target" value="{target}"/>"""

    xml_film = f"""<film type="hdrfilm">
            <integer name="width" value="{xpixel}"/>
            <integer name="height" value="1"/>
            <rfilter type="box"/>
        </film>"""

    return f"""<sensor version="2.0.0" type="distant">
        {xml_direction}
        {xml_target}
        {xml_film}
    </sensor>"""


def make_sensor(direction=None, target=None, xpixel=1):
    from mitsuba.core.xml import load_string
    return load_string(xml_sensor(direction, target, xpixel))


def test_construct(variant_scalar_rgb):
    from mitsuba.core.xml import load_string

    # Construct without parameters (wrong film size)
    with pytest.raises(RuntimeError):
        sensor = load_string("""<sensor version="2.0.0" type="distant"/>""")

    # Construct with wrong film size
    with pytest.raises(RuntimeError):
        sensor = make_sensor(xpixel=2)

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


@pytest.mark.parametrize("sample1, sample2", [
    [[0.32, 0.87], [0.16, 0.44]],
    [[0.17, 0.44], [0.22, 0.81]],
    [[0.12, 0.82], [0.99, 0.42]],
    [[0.72, 0.40], [0.01, 0.61]],
])
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
def test_sample_ray(variant_scalar_rgb, sample1, sample2, direction, target, ray_kind):
    sensor = make_sensor(direction, target)

    if ray_kind == "regular":
        ray, _ = sensor.sample_ray(1., 1., sample1, sample2, True)
    elif ray_kind == "differential":
        ray, _ = sensor.sample_ray_differential(1., 1., sample1, sample2, True)
        assert not ray.has_differentials

    # Check that ray direction is what is expected
    assert ek.allclose(ray.d, ek.normalize(direction))

    # Check that ray origin is outside of bounding sphere
    # Bounding sphere is centered at world origin and has radius 1 without scene
    assert ek.norm(ray.o) >= 1.


def make_scene(direction=[0, 0, -1], target=None):
    from mitsuba.core.xml import load_string

    scene_xml = f"""
    <scene version="2.0.0">
        {xml_sensor(direction, target)}
        <shape type="rectangle"/>
    </scene>
    """

    return load_string(scene_xml)


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
    direction = ek.normalize(direction)
    scene = make_scene(direction=direction)
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
    # around (0, 0, 0)
    isect_valid = isect[~np.isnan(isect).all(axis=1)]
    assert np.allclose(isect_valid[:, :2].mean(axis=0), 0., atol=1e-2)
    assert np.allclose(isect_valid[:, 2], 0., atol=1e-5)

    # Check number of invalid intersections
    # We expect a ratio of invalid interactions equal to the square's area
    # divided by the bounding sphere's cross section, weighted by the surface's
    # slanting factor (cos theta) w.r.t the sensor's direction
    n_invalid = np.count_nonzero(np.isnan(isect).all(axis=1))
    assert np.allclose(n_invalid / n_rays, 1. - 2. / np.pi *
                       ek.dot(direction, [0, 0, -1]), atol=1e-2)


@pytest.mark.parametrize("radiance", [10**x for x in range(-3, 4)])
def test_render(variant_scalar_rgb, radiance):
    # Test render results with a simple scene
    from mitsuba.core.xml import load_string
    import numpy as np

    scene_xml = """
    <scene version="2.0.0">
        <default name="radiance" value="1.0"/>
        <default name="spp" value="1"/>

        <integrator type="path"/>

        <sensor type="distant">
            <film type="hdrfilm">
                <integer name="width" value="1"/>
                <integer name="height" value="1"/>
                <string name="pixel_format" value="rgb"/>
                <rfilter type="box"/>
            </film>

            <sampler type="independent">
                <integer name="sample_count" value="$spp"/>
            </sampler>
        </sensor>

        <emitter type="constant">
            <spectrum name="radiance" value="$radiance"/>
        </emitter>
    </scene>
    """

    scene = load_string(scene_xml, spp=1, radiance=radiance)
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    img = sensor.film().bitmap()
    assert np.allclose(np.array(img), radiance)
