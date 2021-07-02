import mitsuba
import enoki as ek
import numpy as np
import pytest


def make_scene(scene_id, spp=256, film_resolution=(1, 1)):
    from mitsuba.core.xml import load_dict
    from mitsuba.core import ScalarTransform4f

    film_dict = {
        "type": "hdrfilm",
        "width": film_resolution[0],
        "height": film_resolution[1],
        "component_format": "float32",
        "pixel_format": "RGB",
        "rfilter": {"type": "box"},
    }

    sampler_dict = {
        "type": "independent",
        "sample_count": spp,
    }

    sensor_dict = {
        "type": "irradiancemeter",
        "sampler": sampler_dict,
        "film": film_dict,
    }

    if scene_id == "square":
        shape_dicts = {
            "shape": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(2.0),
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "sensor": sensor_dict,
            },
        }
        emitter_dict = {"type": "directional", "direction": [0, 0, -1]}

    elif scene_id == "square_occluded":
        shape_dicts = {
            "square": {
                "type": "rectangle",
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "sensor": sensor_dict,
            },
            "occluder": {
                "type": "rectangle",
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "to_world": ScalarTransform4f.translate([0, 0, 1]),
            },
        }
        emitter_dict = {"type": "directional", "direction": [0, 0, -1]}

    elif scene_id == "sphere_dyson":
        shape_dicts = {
            "sphere": {
                "type": "sphere",
                "flip_normals": True,
                "radius": 2.0,
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "sensor": sensor_dict,
            }
        }
        emitter_dict = {"type": "point", "intensity": 1.0}

    elif scene_id == "sphere":
        shape_dicts = {
            "sphere": {
                "type": "sphere",
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "sensor": sensor_dict,
            }
        }
        emitter_dict = {"type": "directional", "direction": [0, 0, -1]}

    else:
        raise ValueError(f"unsupported scene id '{scene_id}'")

    scene_dict = {
        "type": "scene",
        **shape_dicts,
        "emitter": emitter_dict,
        "integrator": {"type": "pathconnect"},
    }

    return load_dict(scene_dict)


@pytest.mark.parametrize(
    "scene_id, spp, expected",
    [
        ("square", 256, 1.0),  # A simple square illuminated by a directional light
        ("square_occluded", 256, 0.0),  # Same as previous with an occluder
        ("sphere", 10000, 0.25),  # A sphere illuminated by a directional light
        ("sphere_dyson", 256, 0.25),  # A sphere with flipped normal and a point light at its center
    ],
)
def test_sensor_nee(variant_scalar_rgb, scene_id, spp, expected):
    scene = make_scene(scene_id, spp)
    sensor = scene.sensors()[0]

    scene.integrator().render(scene, sensor)
    result = np.array(sensor.film().bitmap()).squeeze()
    rtol = {
        "square": 1e-4,
        "sphere_dyson": 1e-4,
        "sphere": 5e-3,
    }.get(scene_id, 1e-5)
    atol = {
        # "square_occluded": 1e-5,
    }.get(scene_id, 1e-8)
    assert np.allclose(result, expected, rtol=rtol, atol=atol)
