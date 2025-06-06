# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2025-05-06
* Add gz and xml to accepted distribution type
* add multipart argument to from_bytes default to True
* set multipart to false in the size of the file is smaller than the chunk size
* Adding async functionality for some core file system methods.
* raise error immediately when attempting to download dataset you are not subscribed to
* download available format when only one is available and format is set to None
* reduce overhead of to_df when files are already downloaded
* Exception handling improvement
* Limit Licensing service calls
* Limit list datasets calls internally
* Support xml format for distribution type
* Option to disable disc logging 
* Remove the job-lib dependency to fix the credentials generating for each download and upload
* Enable exact search for list datasets
* Limit list catalogs calls internally


## [0.0.1] - 2025-02-04
* first release of python 3.7 compatible pyfusion-3.7 synced with pyfusion version 2.0.6
