## [CMULAB](https://github.com/neulab/cmulab/) extension for [Allosaurus](https://github.com/xinjli/allosaurus/)


### Install

```
python3 -m pip install git+https://github.com/zaidsheikh/cmulab_allosaurus
```

This package registers itself as a plugin for [CMULAB](https://github.com/neulab/cmulab/) (CMU Linguistic Annotation Backend) by [registering an entrypoint](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#dynamic-discovery-of-services-and-plugins) via setuptools.

```
[options.entry_points]
cmulab.plugins =
    allosaurus = cmulab_allosaurus:get_results
```
