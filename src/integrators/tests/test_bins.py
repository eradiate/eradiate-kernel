import numpy as np
import pytest

import mitsuba


def test_construct(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict

    # Construct with complete specification
    i = load_dict({
        "type": "bins",
        "bins": "01:400:500, 02:500:600, 03:600:700, 04:700:800",
        "integrator": {"type": "path"}
    })
    assert i is not None
    assert i.aov_names() == ["01", "01_weights", 
                             "02", "02_weights", 
                             "03", "03_weights", 
                             "04", "04_weights"]

    # Construct with ill-formed bins
    i = load_dict({
        "type": "bins",
        "bins": "01:400:500, 02:500:600, 03:600:700, 04:700:800, oh-no",
        "integrator": {"type": "path"}
    })
    assert i is not None
    assert i.aov_names() == ["01", "01_weights", 
                             "02", "02_weights", 
                             "03", "03_weights", 
                             "04", "04_weights"]

    # No valid bins
    assert load_dict({
        "type": "bins",
        "bins": "oh-no",
        "integrator": {"type": "path"}
    }).aov_names() == []

    # Omit bins
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "bins",
            "integrator": {"type": "path"}
        })

    # Omit integrator
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "bins",
            "bins": "01:400:500, 02:500:600, 03:600:700, 04:700:800",
        })


def scene_dict(spp=100, integrator=None, emitter=None):
    result = {
        "type": "scene",
        "sensor": {
            "type": "radiancemeter",
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"}
            },
            "sampler": {
                "type": "independent",
                "sample_count": spp
            },
            "srf": {
                "type": "uniform",
                "lambda_min": 400.,
                "lambda_max": 800.,
                "value": 1.
            }
        }
    }

    if emitter is not None:
        result["emitter"] = emitter

    if integrator is not None:
        result["integrator"] = integrator

    return result


def test_sample(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict
    from mitsuba.core import Bitmap, Struct

    # WIP: the plugin and these tests don't really do what they should!
    
    # Check that average radiance on each bin is recovered with a 
    # sufficient number of samples.
    # Very important: don't forget to account for the sensor's SRF when 
    # calculating the expected value!
    scene = load_dict(scene_dict(
        integrator={
            "type": "bins",
            "bins": "01:300:500, 02:500:600, 03:600:750",
            "integrator": {"type": "path"}
        },
        emitter={
            "type": "constant",
            "radiance": {
                "type": "irregular",
                "wavelengths": "300, 400, 500, 600, 700, 800",
                "values": "0.0, 0.2, 0.4, 0.6, 0.4, 0.2",
            }
        },
        spp=1000
    ))

    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    film = sensor.film()
    img = np.array(film.bitmap()).squeeze()

    result = img[4::2] / img[5::2]
    print(result)
    expected = [0.3, 0.5, 0.45]
    assert np.allclose(result, expected, rtol=3e-3)
    print(img)

    print(film.bitmap(raw=True))
    img = np.array(film.bitmap(raw=True))
    print(img)
