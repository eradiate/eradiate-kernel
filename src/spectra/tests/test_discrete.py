import pytest

import mitsuba
import enoki as ek


def test_construct(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict

    # Full constructor specification
    assert load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800.",
        "values": "4, 5, 6, 7, 8",
        "pmf": "1, 1, 1, 1, 1"
    }) is not None

    # Reduced PMF value array
    assert load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800.",
        "values": "4, 5, 6, 7, 8",
        "pmf": "1"
    }) is not None

    # No PMF value array
    assert load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800.",
        "values": "4, 5, 6, 7, 8"
    }) is not None

    # Reduced value array
    assert load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800.",
        "values": "5",
    }) is not None

    # Incorrect PMF array size
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "discrete",
            "wavelengths": "400., 500., 600., 700., 800.",
            "values": "5",
            "pmf": "1, 2"
        })

    # Incorrect value array size
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "discrete",
            "wavelengths": "400., 500., 600., 700., 800.",
            "values": "5, 6",
        })

    # Missing wavelength array
    with pytest.raises(RuntimeError):
        load_dict({
            "type": "discrete",
        })

def test_eval_pdf(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict
    from mitsuba.render import SurfaceInteraction3f

    s = load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800."
    })

    si = SurfaceInteraction3f()
    assert ek.allclose(s.eval(si), 0.)
    assert ek.allclose(s.pdf_spectrum(si), 0.)


def test_sample(variant_scalar_spectral):
    from mitsuba.core.xml import load_dict
    from mitsuba.render import SurfaceInteraction3f

    # Equiprobable wavelengths
    s = load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600., 700., 800.",
        "values": "10",
        "pmf": "1"
    })

    sample = [0.1, 0.3, 0.6, 0.9]
    si = SurfaceInteraction3f()
    wavelengths, spectrum = s.sample_spectrum(si, sample)
    assert ek.allclose(wavelengths, [400, 500, 600, 800])
    assert ek.allclose(spectrum, 10)

    # Higher chance to pick a certain wavelength
    s = load_dict({
        "type": "discrete",
        "wavelengths": "400., 500., 600.",
        "values": "1, 2, 3",
        "pmf": "1, 0.5, 0.5"
    })

    sample = [0.1, 0.3, 0.6, 0.9]
    si = SurfaceInteraction3f()
    wavelengths, spectrum = s.sample_spectrum(si, sample)
    assert ek.allclose(wavelengths, [400, 400, 500, 600])
    assert ek.allclose(spectrum, [1, 1, 2, 3])