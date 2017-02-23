# demo

Source code for application demos.

Version numbers follow [Semantic Versioning v2.0.0](http://semver.org/spec/v2.0.0.html). (Version numbers have format `N.N.N[-X]`, where N are integer numbers and X is an optional label string. Git tags of versions have format `vN.N.N[-X]`.)

## Examples

See Jupyter Notebooks in `demo/app_*/examples` directories for examples.

## Installation

`demo` requires [Python 3x](https://www.continuum.io/downloads#_unix).

To download and checkout the current version, clone the repository, add `demo` to the module search path, then import:
```
$ git clone https://github.com/stharrold/demo.git
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'demo'))
>>> import demo
```

To update and checkout a specific version (e.g. v0.0.1), update your local repository then checkout the version's tag:
```
$ cd demo
$ git pull
$ git checkout tags/v0.0.1
$ cd ..
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'demo'))
>>> import demo
>>> demo.__version__
'0.0.1'
```

## Testing

Test within a [Docker](https://www.docker.com/) container built from the [Dockerfile](Dockerfile). See DockerHub repository [stharrold/demo](https://hub.docker.com/r/stharrold/demo/) for pre-built container images. Test using [pytest](http://pytest.org/) from within `demo`. Example:  
```
$ # default working directory within the container is /opt/demo
$ docker run --rm --tty stharrold/demo:v0.0.1_TODO py.test -v
```
