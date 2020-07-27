import enoki as ek
import pytest
import mitsuba
import random


def dict_sensor(orig, dirs, pixels):
    dict = {
        "type": "radiancemeterarray",
        "origins": orig,
        "directions": dirs,
        "film": {
            "type": "hdrfilm",
            "width": pixels,
            "height": 1,
            "rfilter": {
                "type": "box"
            }
        }
    }
    return dict


def dict_scene(origins, directions, width, spp, radiance):

    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "path"
        },
        "sensor": {
            "type": "radiancemeterarray",
            "origins": origins,
            "directions": directions,
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": 1,
                "pixel_format": "rgb",
                "rfilter": {
                    "type": "box"
                }
            },
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            }
        },
        "emitter": {
            "type": "constant",
            "radiance": {
                "type": "uniform",
                "value": radiance
            }
        }
    }
    return scene_dict


def test_instantiation(variant_scalar_rgb):
    from mitsuba.core.xml import load_dict

    # Regular instatiation
    for origins, directions in [
            [["0, 0, 0"],
             ["1, 0, 0"]],
            [["0, 0, 0"] * 2,
             ["1, 0, 0", "-1, 0, 0"]]
    ]:

        sensor = load_dict(dict_sensor(", ".join(origins),
                                       ", ".join(directions), len(origins)))
        assert sensor is not None
        assert not sensor.bbox().valid()  # Degenerate bounding box

    with pytest.raises(RuntimeError):
        sensor_invalid = load_dict(dict_sensor("0, 0, 0", "1, 0, 0", 2))

    with pytest.raises(RuntimeError):
        sensor_invalid = load_dict(dict_sensor("0, 0", "1, 0", 1))

    with pytest.raises(RuntimeError):
        sensor_invalid = load_dict(dict_sensor("0, 0, 0", "1, 0", 1))


def test_ray_sampling(variant_scalar_rgb):
    """Test ray sampling by instantiating a sensor with two components. We sample
    rays with different values for the position sample and assert, that the 
    correct component is picked."""

    from mitsuba.core.xml import load_dict

    sensor = load_dict(dict_sensor("0, 0, 0, 1, 0, 1", "1, 0, 0, 1, 1, 1", 2))

    random.seed(42)
    for i in range(10):
        wavelength_sample = random.random()
        position_sample = (random.random(), random.random())
        ray = sensor.sample_ray_differential(
            0, wavelength_sample, position_sample, (0, 0), True)[0]

        if position_sample[0] < 0.5:
            assert ek.allclose(ray.o, (0., 0., 0.))
            assert ek.allclose(ray.d, ek.normalize((1., 0., 0.)))
        else:
            assert ek.allclose(ray.o, (1., 0., 1.))
            assert ek.allclose(ray.d, ek.normalize((1., 1., 1.)))


def test_many_sensors(variant_scalar_rgb):
    """Test rendering with a large number of radiancemeters"""

    from mitsuba.core import set_thread_count
    from mitsuba.core.xml import load_dict
    import numpy as np
    import time

    num_sensors = 360 * 90
    max_threads = 4
    orig_list = list()
    dir_list = list()
    for i in range(num_sensors):
        num1 = np.random.rand()
        num2 = np.random.rand()
        num3 = np.random.rand()
        orig_list.append(f"{num1}, {num2}, {num3}, ")
        dir_list.append(f"{-num1}, {-num2}, {-num3}, ")

    origins = "".join(orig_list)
    directions = "".join(dir_list)

    scene_dict = dict_scene(origins, directions, num_sensors, 5, 100)

    scene = load_dict(scene_dict)
    sensor = scene.sensors()[0]

    set_thread_count(1)
    start1 = time.time()
    scene.integrator().render(scene, sensor)
    stop1 = time.time()
    time1 = stop1 - start1

    set_thread_count(max_threads)
    start2 = time.time()
    scene.integrator().render(scene, sensor)
    stop2 = time.time()
    time2 = stop2 - start2

    ratio = time1 / time2

    assert np.isclose(ratio, (max_threads), atol=1)


@pytest.mark.parametrize("radiance", [10**x for x in range(-3, 4)])
def test_render(variant_scalar_rgb, radiance):
    # Test render results with a simple scene
    from mitsuba.core.xml import load_dict
    import numpy as np

    scene_dict = dict_scene("1, 0, 0, 0, 1, 0, 0, 0, 1", "1, 0, 0, 0, -1, 0, 0, 0, -1",
                            3, 1, radiance)

    scene = load_dict(scene_dict)
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    img = sensor.film().bitmap()
    assert np.allclose(np.array(img), radiance)


def test_render_complex(variant_scalar_rgb):
    # Test render results for a more complex scene.
    # The radiancemeterarray has three components, pointing at three
    # surfaces with different reflectances.

    from mitsuba.core.xml import load_dict
    from mitsuba.core import ScalarTransform4f, set_thread_count
    import numpy as np

    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "path"
        },
        "sensor": {
            "type": "radiancemeterarray",
            "origins": "-2, 0, 1, 0, 0, 1, 2, 0, 1",
            "directions": "0, 0, -1, 0, 0, -1 0, 0, -1",
            "film": {
                "type": "hdrfilm",
                "width": 3,
                "height": 1,
                "pixel_format": "luminance",
                "rfilter": {
                    "type": "box"
                }
            },
            "sampler": {
                "type": "independent",
                "sample_count": 1,
            }
        },
        "emitter": {
            "type": "directional",
            "direction": (0, 0, -1),
            "irradiance": {
                "type": "uniform",
                "value": 1
            }
        },
        "light_rectangle": {
            "type": "rectangle",
            "to_world": ScalarTransform4f.translate((-2, 0, 0)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": 1.0
                }
            }
        },
        "medium_rectangle": {
            "type": "rectangle",
            "to_world": ScalarTransform4f.translate((0, 0, 0)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": 0.5
                }
            }
        },
        "dark_rectangle": {
            "type": "rectangle",
            "to_world": ScalarTransform4f.translate((2, 0, 0)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": 0.0
                }
            }
        },
    }

    scene = load_dict(scene_dict)
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    img = sensor.film().bitmap()
    data = np.squeeze(np.array(img))

    assert np.isclose(data[0] / data[1], 2, atol=1e-3)
    assert data[2] == 0
