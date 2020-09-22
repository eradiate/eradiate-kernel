import mitsuba
import pytest
import numpy as np
import enoki as ek
import itertools


# Don't change these!
nx = ny = nz = 3
channel_count = 3
DATA = np.zeros((nx, ny, nz, channel_count))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            for l in range(channel_count):
                DATA[i, j, k, l] = i + j + k + l * 5.


@pytest.fixture(scope="session")
def binary_gridvolume_file(tmpdir_factory):
    """Write values in a numpy array to a binary file so that
    a ``gridvolume`` kernel plugin can be instantiated with that file.

    Parameter ``filename`` (str):
        File name.

    Parameter ``values`` (array):
        Values of the extinction coefficients or albedo
    """

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py
    filename = tmpdir_factory.mktemp("textures").join("data.vol")
    global DATA

    with open(filename, "wb") as f:
        f.write(b"V")
        f.write(b"O")
        f.write(b"L")
        f.write(np.uint8(3).tobytes())  # format version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(DATA.shape[0]).tobytes())  # size
        f.write(np.int32(DATA.shape[1]).tobytes())
        f.write(np.int32(DATA.shape[2]).tobytes())
        if DATA.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(DATA.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(DATA.ravel().astype(np.float32).tobytes())

    return filename


def test_construct(binary_gridvolume_file, variant_scalar_spectral):
    from mitsuba.core.xml import load_dict

    t = load_dict({
        "type": "gridvolume_spectral",
        "filename": str(binary_gridvolume_file),
        "lambda_min": 400.,
        "lambda_max": 800.
    }).expand()[0]

    assert t is not None


def test_eval(binary_gridvolume_file, variant_scalar_spectral):
    from mitsuba.core.xml import load_dict
    from mitsuba.render import Interaction3f
    def clamp(val, minv, maxv): return np.maximum(np.minimum(val, maxv), minv)

    lambda_min = 0.
    lambda_max = 1.

    def expected(p, wavelengths):
        global nx, channel_count
        # Convert to bitmap-scaled coordinates (used to compute the expected value)
        p_scaled = clamp(p * nx - 0.5, 0., nx - 1.)
        # Scale to spectral coordinate normalized by array size
        wavelengths_scaled = wavelengths * (channel_count - 1.)
        values = p_scaled.sum() + wavelengths_scaled * 5.

        # Mask out values outside the spectral range
        active_wavelengths = np.logical_and(
            wavelengths >= lambda_min,
            wavelengths <= lambda_max
        )
        result = np.zeros_like(wavelengths)
        result[active_wavelengths] = values[active_wavelengths]

        return result

    t = load_dict({
        "type": "gridvolume_spectral",
        "filename": str(binary_gridvolume_file),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
    }).expand()[0]

    si = Interaction3f()

    for p, wavelengths in itertools.product(
        # Points in local coordinates (the test dataset sets its bounds to [0, 1]^3)
        [
            # A point snapped to a grid point in the middle of the bitmap
            np.array([0.5, 0.5, 0.5]),
            # A point positioned between bitmap samples
            np.array([0.51, 0.51, 0.51]),
            # A point outside of the bitmap grid
            np.array([1, 1, 1]),
            # Another one
            np.array([0, 0, 0]),
        ],
        # Wavelengths in [0, 1] (the covered spectral interval)
        [
            # A point located in the middle of the spectral grid and requiring interpolation
            np.array([0.55, 0.55, 0.55, 0.55]),
            # A point snapped to a grid point in the middle of the array
            np.array([0.5, 0.5, 0.5, 0.5]),
            # A point snapped to the first grid point 
            np.array([0.0, 0.0, 0.0, 0.0]),
            # A point snapped to the last grid point
            np.array([1.0, 1.0, 1.0, 1.0]),
            # A point with wavelengths out of range
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, -1.0, 1.0]),
        ]
    ):
        si.p = p
        si.wavelengths = wavelengths
        values = t.eval(si, True)
        assert ek.allclose(values, expected(p, wavelengths))
