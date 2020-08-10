import mitsuba
import enoki as ek
import pytest
from enoki.dynamic import Float32 as Float


def test_create(variant_scalar_rgb):
    from mitsuba.core import xml

    s = xml.load_dict({"type": "cone"})
    assert s is not None
    assert s.primitive_count() == 1
    assert ek.allclose(s.surface_area(), ek.sqrt(2) * ek.pi)


def test_bbox(variant_scalar_rgb):
    from mitsuba.core import xml, Transform4f, Vector3f

    for h in [1, 5, 10]:
        for r in [1, 2, 4, 8]:
            s = xml.load_dict({
                "type": "cone",
                "to_world": Transform4f.scale((r, r, h))
            })
            b = s.bbox()

            hyp = ek.sqrt(ek.sqr(h) + ek.sqr(r))
            assert ek.allclose(s.surface_area(), ek.pi * r * hyp)
            assert b.valid()
            assert ek.allclose(b.min, -Vector3f([r, r, 0.0]))
            assert ek.allclose(b.max, Vector3f([r, r, h]))


def test_ray_intersect(variant_scalar_rgb):
    from mitsuba.core import xml, Ray3f, Transform4f
    from mitsuba.render import HitComputeFlags

    for r in [1, 2, 4, 8]:
        for l in [1, 5, 10]:
            s = xml.load_dict({
                "type": "scene",
                "foo": {
                    "type": "cone",
                    "to_world": Transform4f.scale((r, r, l))
                }
            })

            # grid size
            n = 10
            for x in ek.linspace(Float, -1, 1, n):
                for z in ek.linspace(Float, -1, 1, n):
                    x = 1.1 * r * x
                    z = 1.1 * l * z

                    ray = Ray3f(o=[x, -10, z], d=[0, 1, 0],
                                time=0.0, wavelengths=[])
                    si_found = s.ray_test(ray)
                    si = s.ray_intersect(
                        ray, HitComputeFlags.All | HitComputeFlags.dNSdUV)

                    assert si_found == si.is_valid()
                    r_h = ((l - z) / l) * r
                    assert si_found == ek.allclose(
                        si.p[0]**2 + si.p[1]**2, r_h**2, atol=2e-2)

                    if si_found:
                        ray_u = Ray3f(ray)
                        ray_v = Ray3f(ray)
                        eps = 1e-4
                        ray_u.o += si.dp_du * eps
                        ray_v.o += si.dp_dv * eps
                        si_u = s.ray_intersect(ray_u)
                        si_v = s.ray_intersect(ray_v)

                        if si_u.is_valid():
                            dp_du = (si_u.p - si.p) / eps
                            dn_du = (si_u.n - si.n) / eps
                            assert ek.allclose(dp_du, si.dp_du, atol=5e-2)
                            assert ek.allclose(dn_du, si.dn_du, atol=5e-2)
                        if si_v.is_valid():
                            dp_dv = (si_v.p - si.p) / eps
                            dn_dv = (si_v.n - si.n) / eps
                            assert ek.allclose(dp_dv, si.dp_dv, atol=5e-2)
                            assert ek.allclose(dn_dv, si.dn_dv, atol=5e-2)
