#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-distant:

Distant directional sensor (:monosp:`distant`)
----------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Sensor-to-world transformation matrix.
 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction from which the
     sensor will be recording in world coordinates.
 * - origin
   - |string|
   - *Optional.* Specify the type of origining. Two types of origin can be specified.
     Setting the parameter to 'rectangle', will lead to a rectangular origin area 
     defined by the  parameters 'origin_min' and 'origin_max'.
     Setting it to 'circle' will produce a circular origin area, defined by 
     the 'origin_center' and 'radius' parameters. To define a point like 
     origin, choose 'circle' and set 'radius' to 0.
     If unset, rays will be cast uniformly over the entire scene.
 * - origin_a
   - |point|
   - Coordinates in the xy-plane for the rectangular origin area.
 * - origin_b
   - |point|
   - Maximum coordinates in the xy-plane for the rectangular origin area.
 * - origin_center
   - |point|
   - Center point in the xy-plane for the circular origin area.
 * - origin_radius
   - |float|
   - Radius for the circular origin area
    

This sensor plugin implements a distant directional sensor which records
radiation leaving the scene in a given direction. By default, it records the 
(spectral) radiant flux per unit solid angle leaving the scene in the specified
direction (in unit power per unit solid angle). Rays cast by the sensor are 
distributed uniformly on the cross section of the scene's bounding sphere.

.. warning::

    If this sensor is used with an environment map emitter, it will also record 
    radiant flux coming from the part of emitter appearing through the scene's 
    bounding sphere cross section. Care should be taken notably when using the
    `constant` or `envmap` emitters.

If the ``origin`` parameter is set, the sensor looks at a single point and
records a (spectral) radiant flux per unit surface area per unit solid angle 
(in unit power per unit surface area per unit solid angle).

With the ``origin_`` parameters users can define an area parallel to the xy-plane
that the sensor's rays will originate from. The area can be defined in two ways:

- Setting `origin` to 'rectangle' lets users define a rectangular area through 
  two points `origin_a` and `origin_b`. The resulting area will cover the area 
  from the smaller to the larger value in both dimensions.
- Setting `origin` to 'circle'  lets users define a circular origin area 
  through a center `origin_center` and a radius `origin_radius`.
- If `origin` is not set, the sensor will origin the entire cross section
  of the scene's bounding sphere.

*/

MTS_VARIANT class DistantSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES(Scene)

    DistantSensor(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(normalize(props.vector3f("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_world_transform =
                new AnimatedTransform(ScalarTransform4f::look_at(
                    ScalarPoint3f(0.0f), ScalarPoint3f(direction), up));
        }

        if (props.has_property("origin")) {
            m_origin     = props.string("origin", "circle");
        } else {
            m_origin = std::string("None");
            m_origin_type = 0;
        }

        m_origin_a = props.point3f("origin_a", 0.f);
        m_origin_b = props.point3f("origin_b", 0.f);
        m_origin_center = props.point3f("origin_center", 0.f);
        m_origin_radius = props.float_("origin_radius", 0.f);

        if (m_origin == "rectangle") {
            if (!props.has_property("origin_a") || !props.has_property("origin_b")) {
                Throw("Rectangular origin requires the 'origin_a' and 'origin_b' parameters");
            }
            if (!m_origin_a.z() == m_origin_b.z()) {
                Throw("z-components of origin_a and origin_b do not match. "
                      "Cannot determine origin zone elevation.");
            }
            m_origin_type = 1;
            m_origin_area = abs(m_origin_b.x()-m_origin_a.x()) * abs(m_origin_b.y()-m_origin_a.y());
        } else if (m_origin == "circle") {
            if (!props.has_property("origin_center") || !props.has_property("origin_radius")) {
                Throw("Circular origin requires the 'origin_center' and 'origin_radius' parameters");
            }
            m_origin_type = 2;
            m_origin_area = math::Pi<Float> * sqr(m_origin_radius);
        }

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f & /*film_sample*/,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        Float origin_area = 0.f;
        if (m_origin_type == 0) {
            // If no origin is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else if (m_origin_type == 1) {
            Float origin_x = abs(m_origin_b.x() - m_origin_a.x()) * aperture_sample.x() + min(m_origin_a.x(), m_origin_b.x());
            Float origin_y = abs(m_origin_b.y() - m_origin_a.y()) * aperture_sample.y() + min(m_origin_a.y(), m_origin_b.y());
            ray.o = Point3f{origin_x, origin_y, m_origin_a.z()};
        } else if (m_origin_type == 2) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_origin_radius;
            Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_origin_center;
            ray.o = Point3f{sample_disk.x(), sample_disk.y(), m_origin_center.z()};
        }

        ray.update();
        return std::make_pair(
            ray, m_has_origin
                     ? wav_weight
                     : wav_weight * m_origin_area * Frame3f::cos_theta(ray.d));
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f & /*film_sample*/,
        const Point2f &aperture_sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        Float origin_area = 0.f;
        if (m_origin_type == 0) {
            // If no origin  is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else if (m_origin_type == 1) {
            Float origin_x = abs(m_origin_b.x() - m_origin_a.x()) * aperture_sample.x() + min(m_origin_a.x(), m_origin_b.x());
            Float origin_y = abs(m_origin_b.y() - m_origin_a.y()) * aperture_sample.y() + min(m_origin_a.y(), m_origin_b.y());
            ray.o = Point3f{origin_x, origin_y, m_origin_a.z()};
        } else if (m_origin_type == 2) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_origin_radius;
            Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_origin_center;
            ray.o = Point3f{sample_disk.x(), sample_disk.y(), m_origin_center.z()};
        }

        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        ray.update();
        return std::make_pair(
            ray, m_has_origin
                     ? wav_weight
                     : wav_weight * m_origin_area * Frame3f::cos_theta(ray.d));
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantSensor[" << std::endl
            << "  world_transform = " << m_world_transform << "," << std::endl
            << "  bsphere = " << m_bsphere << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    std::string m_origin;
    bool m_has_origin;
    Point3f m_origin_a;
    Point3f m_origin_b;
    Point3f m_origin_center;
    Float m_origin_radius;
    Int32 m_origin_type;
    Float m_origin_area;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantSensor, "DistantSensor");
NAMESPACE_END(mitsuba)
