# Preprocessing Toolbox

![GitHub issues](https://img.shields.io/github/issues/environmental-forecasting/preprocess-toolbox?style=plastic)
![GitHub closed issues](https://img.shields.io/github/issues-closed/environmental-forecasting/preprocess-toolbox?style=plastic)
![GitHub](https://img.shields.io/github/license/environmental-forecasting/preprocess-toolbox)
![GitHub forks](https://img.shields.io/github/forks/environmental-forecasting/preprocess-toolbox?style=social)
![GitHub forks](https://img.shields.io/github/stars/environmental-forecasting/preprocess-toolbox?style=social)

This is the preprocessing library for taking `download-toolbox` datasets and
combining / composing multi-source data loaders that can be used to cache or supply downstream applications.

This is only just getting started, more info will appear soon.

Contact `jambyr <at> bas <dot> ac <dot> uk` if you want further information.

## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Implementation](#implementation)
* [Contributing](#contributing)
* [Credits](#credits)
* [License](#license)

## Installation

Not currently released to pip.

Please refer to [the contribution guidelines for more information.](CONTRIBUTING.md)

## Implementation

When installed, the library will provide a series of CLI commands. Please use
the `--help` switch for more initial information, or the documentation.

### Basic principles

The library provides the ability to preprocess `download-toolbox` datasets and create singular configurations for reading out the data in a multi-channel format for dataset construction:

1. Preprocess datasets from download-toolbox so that the dataset is continuous and normalised for the downstream application
1. Generate a loader configuration, applying additional metadata (arbitrary channels and masks) providing initial access to the collected data
1. Use this data loader to produce usable application datasets for downstream applications (testing with IceNet and another internal application)

This is a base library upon which application specific processing is based, lowering the implementation overhead for creating multi-source datasets for environmental applications that require integration of data from sources that `download-toolbox` provide.

This library doesn't have knowledge of those datasets, it forms the basis for processing things specific to an application by importing application-specific logic dynamically. See [this issue](https://github.com/environmental-forecasting/preprocess-toolbox/issues/1) for a quick idea of how this works with the [IceNet workflow](https://github.com/icenet-ai/icenet).

## Limitations

**This is currently active development, but the commands work to most degrees.**

## Contributing

Please refer to [the contribution guidelines for more information.](CONTRIBUTING.md)

## Credits

<a href="https://github.com/environmental-forecasting/preprocess-toolbox/graphs/contributors"><img src="https://contrib.rocks/image?repo=environmental-forecasting/preprocess-toolbox" /></a>

## License

This is licensed using the [MIT License](LICENSE)
