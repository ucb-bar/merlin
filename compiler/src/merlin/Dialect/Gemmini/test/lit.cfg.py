import lit.formats

config.name = "MerlinGemmini"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
