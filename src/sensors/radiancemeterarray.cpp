#include <enoki/array_router.h>
#include <enoki/array_traits.h>
#include <enoki/fwd.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-radiancemeterarray:

Radiance meter (:monosp:`radiancemeterarray`)
---------------------------------------------

.. pluginparameters::

 * - origins
   - |string|
   - Comma separated list of locations from which the sensors will be recording
     in world coordinates.

 * - directions
   - |string|
   - Comma separated list of directions in which the sensors are pointing in
     world coordinates.

This sensor plugin implements multiple radiance meters, as implemented in the
:monosp:`radiancemeter` plugin.

This sensor allows using the inherent parallelization of Mitsuba2, which is not
possible with the :monosp.`radiancemeter` due to its film size of 1x1.

The origin points and direction vectors for this sensor are specified as a list
of floating point values, where three subsequent values will be grouped into a
point or vector respectively.

The following snippet shows how to specify a :monosp:`radiancemeterarray` with
two sensors, one located ad (1, 0, 0) and pointing in the direction (-1, 0, 0),
the other located at (0, 1, 0) and pointing in the direction (0, -1, 0).

.. code-block:: xml

    <sensor type="radiancemeterarray">
            <string name="origins" value="1, 0, 0, 0, 1, 0"/>
            <string name="directions" value="-1, 0, 0, 0, -1, 0/>
    </sensor>

*/

MTS_VARIANT class RadianceMeterArray final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film, m_world_transform, m_needs_sample_2,
                    m_needs_sample_3)
    MTS_IMPORT_TYPES()

    using Matrix           = enoki::Matrix<Float, Transform4f::Size>;
    using TransformStorage = DynamicBuffer<Matrix>;
    using Index            = int32_array_t<Float>;

    RadianceMeterArray(const Properties &props) : Base(props) {
        if (props.has_property("to_world")) {
            Throw("This sensor is specified through a set of origin and "
                  "direction "
                  "values and cannot use the to_world transform.");
        }

        std::vector<std::string> origins_str =
            string::tokenize(props.string("origins"), " ,");
        std::vector<std::string> directions_str =
            string::tokenize(props.string("directions"), " ,");

        if (origins_str.size() % 3 != 0)
            Throw("Invalid specification! Number of parameters %s, is not a "
                  "multiple of three.",
                  origins_str.size());

        if (origins_str.size() != directions_str.size())
            Throw("Invalid specification! Number of parameters for origins and "
                  "directions (%s, %s) "
                  "are not equal.",
                  origins_str.size(), directions_str.size());

        m_sensor_count = origins_str.size() / 3.;
        m_transforms   = empty<TransformStorage>(m_sensor_count * 16);
        m_transforms.managed();

        for (size_t i = 0; i < m_sensor_count; ++i) {
            size_t index = i * 3;
            ScalarPoint3f origin =
                ScalarPoint3f(std::stof(origins_str[index]),
                              std::stof(origins_str[index + 1]),
                              std::stof(origins_str[index + 2]));

            ScalarVector3f direction =
                ScalarVector3f(std::stof(directions_str[index]),
                               std::stof(directions_str[index + 1]),
                               std::stof(directions_str[index + 2]));

            ScalarPoint3f target = origin + direction;
            auto [up, unused]    = coordinate_system(direction);
            auto transform =
                ScalarTransform4f::look_at(origin, target, up).matrix;
            memcpy(&m_transforms[i * 16], &transform, sizeof(ScalarFloat) * 16);
        }

        if (m_film->size() != ScalarPoint2i(m_transforms.size() / 16, 1))
            Throw("Film size must match the number of radiancemeters defined."
                  "Found: %s vs. %s",
                  m_film->size(), m_transforms.size());

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        m_needs_sample_2 = true;
        m_needs_sample_3 = false;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray origin and direction
        Int32 sensor_index(position_sample.x() * m_sensor_count);
        Index index(sensor_index);

        Matrix coefficients = gather<Matrix>(m_transforms, index);
        Transform4f trafo(coefficients);
        ray.o = trafo.transform_affine(Point3f{ 0.f, 0.f, 0.f });
        ray.d = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &position_sample,
        const Point2f & /*aperture_sample*/, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray origin and direction
        Int32 sensor_index(position_sample.x() * m_sensor_count);
        Index index(sensor_index);

        Matrix coefficients = gather<Matrix>(m_transforms, index);
        Transform4f trafo(coefficients);
        ray.o = trafo.transform_affine(Point3f{ 0.f, 0.f, 0.f });
        ray.d = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Set differentials; since we treat the pixels as individual
        // sensors, we don't have differentials
        ray.has_differentials = false;

        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    ScalarBoundingBox3f bbox() const override {
        // Return an invalid bounding box
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RadianceMeterArray[" << std::endl
            << "  transforms = " << m_transforms << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    TransformStorage m_transforms;
    size_t m_sensor_count;
};

MTS_IMPLEMENT_CLASS_VARIANT(RadianceMeterArray, Sensor)
MTS_EXPORT_PLUGIN(RadianceMeterArray, "RadianceMeterArray");

NAMESPACE_END(mitsuba)
