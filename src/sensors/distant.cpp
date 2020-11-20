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
 * - target
   - |string|
   - *Optional.* Specify the type of targeting. Two types of target can be specified.
     Setting the parameter to 'rectangle', will lead to a rectangular target area 
     defined by the  parameters 'target_min' and 'target_max'.
     Setting it to 'circle' will produce a circular target area, defined by 
     the 'target_center' and 'radius' parameters. To define a point like 
     target, choose 'circle' and set 'radius' to 0.
     If unset, rays will be cast uniformly over the entire scene.
 * - target_min
   - |point|
   - Minimum coordinates in the xy-plane for the rectangular target area.
 * - target_max
   - |point|
   - Maximum coordinates in the xy-plane for the rectangular target area.
 * - target_center
   - |point|
   - Center point in the xy-plane for the circular target area.
 * - target_radius
   - |float|
   - Radius for the circular target area
    

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

If the ``target`` parameter is set, the sensor looks at a single point and
records a (spectral) radiant flux per unit surface area per unit solid angle 
(in unit power per unit surface area per unit solid angle).

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

        if (props.has_property("target")) {
            m_target     = props.string("target", "sphere");
            m_has_target = true;
        } else {
            m_target = std::string("None");
            m_has_target = false;
        }

        m_target_min = props.point3f("target_min", 0.f);
        m_target_max = props.point3f("target_max", 0.f);
        m_target_center = props.point3f("target_center", 0.f);
        m_target_radius = props.float_("target_radius", 0.f);

        if (m_target == "rectangle") {
            if (not props.has_property("target_min") || not props.has_property("target_max")) {
                Throw("Rectangular target requires the 'target_min' and 'target_max' parameters");
            }
        } else if (m_target == "circle") {
            if (not props.has_property("target_center") || not props.has_property("target_radius")) {
                Throw("Circular target requires the 'target_center' and 'target_radius' parameters");
            }
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
        Float target_area = 0.f;
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else {
            if (m_target == "rectangle") {
                Float target_x = (m_target_max.x() - m_target_min.x()) * aperture_sample.x() + m_target_min.x();
                Float target_y = (m_target_max.y() - m_target_min.y()) * aperture_sample.y() + m_target_min.y();
                Point3f target = Point3f{target_x, target_y, 0};
                ray.o = target - 2.f * ray.d * m_bsphere.radius;
                Float target_area = m_target_max.x()-m_target_max.x() * m_target_max.y()-m_target_max.y();
            } else if (m_target == "circle") {
                Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_target_radius;
                Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_target_center;
                Point3f target = Point3f{sample_disk.x(), sample_disk.y(), 0};
                ray.o = target - 2.f * ray.d * m_bsphere.radius;
                Float target_area = math::Pi<Float> * sqr(m_target_radius);
            }
        }

        ray.update();
        return std::make_pair(
            ray, m_has_target
                     ? wav_weight
                     : wav_weight * target_area);
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
        Float target_area = 0.f;
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else {
            if (m_target == "rectangle") {
                Float target_x = (m_target_max.x() - m_target_min.x()) * aperture_sample.x() + m_target_min.x();
                Float target_y = (m_target_max.y() - m_target_min.y()) * aperture_sample.y() + m_target_min.y();
                Point3f target = Point3f{target_x, target_y, 0};
                ray.o = target - 2.f * ray.d * m_bsphere.radius;
                target_area = m_target_max.x()-m_target_max.x() * m_target_max.y()-m_target_max.y();
            } else if (m_target == "circle") {
                Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_target_radius;
                Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_target_center;
                Point3f target = Point3f{sample_disk.x(), sample_disk.y(), 0};
                ray.o = target - 2.f * ray.d * m_bsphere.radius;
                target_area = math::Pi<Float> * sqr(m_target_radius);
            }
        }

        // // 3. Sample ray origin
        // if (!m_has_target) {
        //     // If no target point is defined, sample a target point on the
        //     // bounding sphere's cross section
        //     Point2f offset =
        //         warp::square_to_uniform_disk_concentric(aperture_sample);
        //     Vector3f perp_offset =
        //         trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
        //     ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        // } else {
        //     ray.o = m_target - 2.f * ray.d * m_bsphere.radius;
        // }

        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        ray.update();
        return std::make_pair(
            ray, m_has_target
                     ? wav_weight
                     : wav_weight * target_area);
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
    std::string m_target;
    bool m_has_target;
    Point3f m_target_min;
    Point3f m_target_max;
    Point3f m_target_center;
    Float m_target_radius;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantSensor, "DistantSensor");
NAMESPACE_END(mitsuba)
