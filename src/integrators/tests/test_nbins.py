import warnings

import numpy as np
import pytest

import mitsuba


def test_construct(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict

    # Construct with complete specification
    assert load_dict({
        "type": "nbins",
        "wavelengths": "400, 500, 600, 700",
        "tolerance": 1e-3,
        "integrator": {"type": "path"}
    }) is not None

    # Default tolerance
    assert load_dict({
        "type": "nbins",
        "wavelengths": "400, 500, 600, 700",
        "integrator": {"type": "path"}
    }) is not None

    # Omit wavelengths
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "nbins",
            "integrator": {"type": "path"}
        })

    # Omit integrator
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "nbins",
            "wavelengths": "400, 500, 600, 700",
        })


def integrator_dict(wavelengths, tolerance=None):
    if not isinstance(wavelengths, str):
        wavelengths = ", ".join(map(str, wavelengths))

    d = {
        "type": "nbins",
        "wavelengths": wavelengths,
        "integrator": {"type": "path"}
    }

    if tolerance is not None:
        d["tolerance"] = tolerance

    return d


def scene_dict(wavelengths, spp=100, integrator=None, radiance=1.0):
    if not isinstance(wavelengths, str):
        wavelengths = ", ".join(map(str, wavelengths))

    result = {
        "type": "scene",
        "emitter": {
            "type": "constant",
            "radiance": {
                "type": "uniform",
                "value": radiance
            }
        },
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
                "type": "discrete",
                "wavelengths": wavelengths
            }
        }
    }

    if integrator is not None:
        result["integrator"] = integrator

    return result


def test_sample(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict
    from mitsuba.core import Bitmap, Struct

    def run(scene):
        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)
        film = sensor.film()
        img = np.array(film.bitmap()).squeeze()
        bin_values = img[4::2]
        bin_population = img[5::2]

        return bin_values / bin_population

    lambda_min, lambda_max = 400., 800.

    # Test with as many wavelengths as the number of channels
    radiance = 1
    n_bins = 4
    bin_wavelengths = np.linspace(lambda_min, lambda_max, n_bins)
    scene = load_dict(scene_dict(
        bin_wavelengths,
        integrator=integrator_dict(wavelengths=bin_wavelengths),
        spp=10,
        radiance=radiance
    ))
    assert np.allclose(run(scene), radiance)

    # Test with more wavelengths than channels
    radiance = 1e3
    n_bins = 25
    bin_wavelengths = np.linspace(lambda_min, lambda_max, n_bins)
    scene = load_dict(scene_dict(
        bin_wavelengths,
        integrator=integrator_dict(wavelengths=bin_wavelengths),
        spp=100,
        radiance=radiance
    ))
    assert np.allclose(run(scene), radiance)

    # Test with fewer wavelengths than channels
    radiance = 1e-3
    n_bins = 2
    bin_wavelengths = np.linspace(lambda_min, lambda_max, n_bins)

    scene = load_dict(scene_dict(
        bin_wavelengths,
        integrator=integrator_dict(wavelengths=bin_wavelengths),
        spp=10,
        radiance=radiance
    ))
    assert np.allclose(run(scene), radiance)

    # Test with very few samples (should lead to some bins being unpopulated)
    radiance = 1
    n_bins = 10
    bin_wavelengths = np.linspace(lambda_min, lambda_max, n_bins)

    scene = load_dict(scene_dict(
        bin_wavelengths,
        integrator=integrator_dict(wavelengths=bin_wavelengths),
        spp=1,
        radiance=radiance
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(scene)
    # We expect divide-by-zero problems with unpopulated bins
    assert any(np.isnan(result))
