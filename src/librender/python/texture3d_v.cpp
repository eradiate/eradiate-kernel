#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(Volume) {
    MTS_PY_IMPORT_TYPES(Volume)
    MTS_PY_CLASS(Volume, Object)
        .def("eval",
            vectorize(&Volume::eval),
            "it"_a, "active"_a = true, D(Volume, eval))
        .def("eval_1",
            vectorize(&Volume::eval_1),
            "it"_a, "active"_a = true, D(Volume, eval_1))
        .def("eval_3",
            vectorize(&Volume::eval_3),
            "it"_a, "active"_a = true, D(Volume, eval_3))
        .def("eval_gradient",
            vectorize(&Volume::eval_gradient),
            "it"_a, "active"_a = true, D(Volume, eval_gradient))
        .def("max",
            &Volume::max,
            D(Volume, max))
        .def("bbox",
            &Volume::bbox,
            D(Volume, bbox))
        .def("resolution",
            &Volume::resolution,
            D(Volume, resolution));
}
