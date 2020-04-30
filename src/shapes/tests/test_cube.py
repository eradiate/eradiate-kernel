import pytest

import enoki as ek
import mitsuba
from enoki.dynamic import Float32 as Float


def example_cube(translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0, 0)):
    from mitsuba.core.xml import load_string

    return load_string(
        """<shape version="2.0.0" type="cube">
            <transform name="to_world">
                <scale value="{}, {}, {}"/>
                <rotate value="{}, {}, {}" angle="{}"/>
                <translate x="{}" y="{}" z="{}"/>
            </transform>
        </shape>
        """.format(scale[0], scale[1], scale[2],
                   rotate[0], rotate[1], rotate[2], rotate[3],
                   translate[0], translate[1], translate[2])
    )


def test_create(variant_scalar_rgb):
    # Test plugin instantiation
    cube = example_cube()
    assert cube is not None
    assert ek.allclose(cube.surface_area(), 24.)


def test_bbox(variant_scalar_rgb):
    # Check bounding box definition
    # First without a Transform
    cube1 = example_cube()
    assert cube1 is not None
    bbox = cube1.bbox()
    assert bbox.valid()
    assert ek.allclose(bbox.min, [-1, -1, -1])
    assert ek.allclose(bbox.max, [1, 1, 1])

    # Then with a transform
    cube2 = example_cube(translate=(1, 2, 3))
    assert cube2 is not None
    bbox = cube2.bbox()
    assert bbox.valid()
    assert ek.allclose(bbox.min, [0, 1, 2])
    assert ek.allclose(bbox.max, [2, 3, 4])

    cube3 = example_cube(scale=(1, 2, 3))
    bbox = cube3.bbox()
    assert bbox.valid()
    assert ek.allclose(bbox.min, [-1, -2, -3])
    assert ek.allclose(bbox.max, [1, 2, 3])

    cube4 = example_cube(rotate=(0, 0, 1, 45))
    bbox = cube4.bbox()
    assert bbox.valid()
    assert ek.allclose(bbox.min, [-1.41421, -1.41421, -1])
    assert ek.allclose(bbox.max, [1.41421, 1.41421, 1])

    cube5 = example_cube(translate=(1, 0, 0), scale=(2, 2, 2))
    bbox = cube5.bbox()
    assert bbox.valid()
    assert ek.allclose(bbox.min, [-1, -2, -2])
    assert ek.allclose(bbox.max, [3, 2, 2])
