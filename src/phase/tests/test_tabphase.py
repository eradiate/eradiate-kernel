from itertools import product

import enoki as ek
import numpy as np


def test_create(variant_scalar_rgb):
    from mitsuba.core.xml import load_dict

    p = load_dict({"type": "tabphase", "values": "0.5, 1.0, 1.5"})
    assert p is not None


def test_eval(variant_scalar_rgb):
    """
    Compare eval() output with a reference implementation written in Python.
    We make sure that the values we use to initialize the plugin are such that
    the phase function has an asymmetric lobe.
    """
    from mitsuba.core.xml import load_dict
    from mitsuba.render import MediumInteraction3f, PhaseFunctionContext

    # Phase function table definition
    ref_y = np.array([0.5, 1.0, 1.5])
    ref_x = np.linspace(-1, 1, len(ref_y))
    ref_integral = np.trapz(ref_y, ref_x)

    def eval(wi, wo):
        # Python implementation used as a reference
        wi = wi.reshape((-1, 3))
        wo = wo.reshape((-1, 3))

        if wi.shape[0] == 1:
            wi = np.broadcast_to(wi, wo.shape)
        if wo.shape[0] == 1:
            wo = np.broadcast_to(wo, wi.shape)

        cos_theta = np.array([np.dot(a, b) for a, b in zip(wi, wo)])
        return 0.5 / np.pi * np.interp(-cos_theta, ref_x, ref_y) / ref_integral

    wi = np.array([[0, 0, -1]])
    thetas = np.linspace(0, np.pi / 2, 16)
    phis = np.linspace(0, np.pi, 16)
    wos = np.array(
        [
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
            for theta, phi in product(thetas, phis)
        ]
    )
    ref_eval = eval(wi, wos)

    # Evaluate Mitsuba implementation
    tab = load_dict({"type": "tabphase", "values": ", ".join([str(x) for x in ref_y])})
    ctx = PhaseFunctionContext(None)
    mi = MediumInteraction3f()
    mi.wi = wi.squeeze()
    tab_eval = np.zeros_like(ref_eval)
    for i, wo in enumerate(wos):
        tab_eval[i] = tab.eval(ctx, mi, wo)

    # Compare reference and plugin outputs
    assert np.allclose(ref_eval, tab_eval)


def test_chi2(variant_packet_rgb):
    from mitsuba.python.chi2 import ChiSquareTest, PhaseFunctionAdapter, SphericalDomain

    sample_func, pdf_func = PhaseFunctionAdapter(
        "tabphase", "<string name='values' value='0.5, 1.0, 1.5'/>"
    )

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    result = chi2.run(0.1)
    chi2._dump_tables()
    assert result


def test_traverse(variant_scalar_rgb):
    from mitsuba.core.math import InvTwoPi
    from mitsuba.core.xml import load_dict
    from mitsuba.python.util import traverse
    from mitsuba.render import MediumInteraction3f, PhaseFunctionContext

    # Phase function table definition
    ref_y = np.array([0.5, 1.0, 1.5])
    ref_x = np.linspace(-1, 1, len(ref_y))
    ref_integral = np.trapz(ref_y, ref_x)

    # Initialise as isotropic and update with parameters
    phase = load_dict({"type": "tabphase", "values": "1, 1, 1"})
    params = traverse(phase)
    params["values"] = [0.5, 1.0, 1.5]
    params.update()

    # Distribution parameters are updated
    params = traverse(phase)
    assert ek.allclose(params["values"], [0.5, 1.0, 1.5])

    # The plugin itself evaluates consistently
    ctx = PhaseFunctionContext(None)
    mi = MediumInteraction3f()
    mi.wi = np.array([0, 0, -1])
    wo = [0, 0, 1]
    assert ek.allclose(phase.eval(ctx, mi, wo), InvTwoPi * 1.5 / ref_integral)


def test_compare(variant_scalar_rgb):
    # Compare with another phase function to check for frame orientation issues
    from mitsuba.core import Vector3f
    from mitsuba.core.xml import load_dict
    from mitsuba.render import PhaseFunctionContext, MediumInteraction3f

    # Evaluate Henyey-Greenstein phase function and generate the lookup table
    wi = np.array([[0, 0, -1]])
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 201)
    phis = np.atleast_1d(0.)
    wos = np.array(
        [
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
            for theta, phi in product(thetas, phis)
        ]
    )

    hg = load_dict({"type": "hg", "g": 0.2})
    wo = Vector3f(wos)
    print(wo)
    ctx = PhaseFunctionContext(None)
    mi = MediumInteraction3f()
    tab_values = hg.eval(ctx, mi, wo)
    print(tab_values)
    for i, wo in enumerate(list(wos)):
        tab_eval[i] = tab.eval(ctx, mi, wo)



    # # Generate directions
    # wi = np.array([[0, 0, -1]])
    # thetas = np.linspace(0, np.pi / 2, 16)
    # phis = np.linspace(0, np.pi, 16)
    # wos = np.array(
    #     [
    #         (
    #             np.sin(theta) * np.cos(phi),
    #             np.sin(theta) * np.sin(phi),
    #             np.cos(theta),
    #         )
    #         for theta, phi in product(thetas, phis)
    #     ]
    # )


    # tab = load_dict({"type": "tabphase", "values": ", ".join([str(x) for x in ref_y])})
    # ctx = PhaseFunctionContext(None)
    # mi = MediumInteraction3f()
    # mi.wi = wi.squeeze()
    # tab_eval = np.zeros_like(ref_eval)
    # for i, wo in enumerate(wos):
    #     tab_eval[i] = tab.eval(ctx, mi, wo)

    # # Compare reference and plugin outputs
    # assert np.allclose(ref_eval, tab_eval)
