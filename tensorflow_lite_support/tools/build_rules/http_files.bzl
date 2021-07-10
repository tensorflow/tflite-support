"""Build rule to depend on files downloaded from http_file."""

def tflite_model(name):
    """Links the tflite model from http_file with the current directory.

    Args:
      name: the name of the tflite_model target, which is also the name of the
      tflite model specified through http_file in WORKSPACE. For exmaple, if
      `name` is Foo, `tflite_model` will create a link to the downloaded model
      file "@Foo//file" to the current directory as "Foo.tflite".
    """
    native.genrule(
        name = "%s_ln" % (name),
        srcs = ["@%s//file" % (name)],
        outs = ["%s.tflite" % (name)],
        output_to_bindir = 1,
        cmd = "ln $< $@",
    )

    native.filegroup(
        name = name,
        srcs = ["%s.tflite" % (name)],
    )
