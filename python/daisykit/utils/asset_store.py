# Copyright 2021 The DaisyKit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source file was taken from ncnn library with some modifications.

"""Asset store which provides pretrained assets."""
from __future__ import print_function
import pathlib

__all__ = ["get_asset_file", "purge"]

import os
import zipfile
import logging
import portalocker

from .download import download, check_sha1

_asset_sha1 = {
    name: checksum
    for checksum, name in [('069623b785bc2a0850bce39f0f5ef48a2c37f758', 'images/background.jpg'), ('adee101923317578dc8fa41ca680fd9c6f877187', 'models/human_matting/erd/erdnet.param'), ('c54b251a44fb0e542ff77ea3bac79883a63bc8d7', 'models/human_matting/erd/erdnet.bin'), ('3e57a3fbb29f530411c5fe566c5d8215b05b1913', 'models/facial_landmark/pfld-sim.param'), ('cf8cc7a623ba31cb8c700d59bf7bd622f081d695', 'models/facial_landmark/pfld-sim.bin'), ('b673647bd428ee5eb8b2467f2986a7f3c6b85a7e', 'models/face_detection/slim/slim_320.param'), ('7865fa597b1d659b35070862512a3862c9b7bd6c', 'models/face_detection/slim/slim_320.bin'), ('1ab6d93570e408e8c85c759ee1c61d67cb9509b6', 'models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.bin'), ('7451cb1978b7f51a2cc92248e9c103fd456e0f74', 'models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.param'), ('37271c3e4dbdb21e7fe1939e7b46ff88fc978dba', 'models/face_detection/rfb/RFB-320.param'), ('91ae3956c62b14176e224d0997b8f693f9913612', 'models/face_detection/rfb/RFB-320.bin'), ('88d4e03b3cccc25de4f214ef55d42b52173653c0', 'models/human_detection/ssd_mobilenetv2.bin'), ('c4a952480ea26c1e2097e80f5589df707f31463c', 'models/human_detection/ssd_mobilenetv2.param'), ('ed2c3819ea314f79613cae2cc4edbe509d274702', 'models/action_classification/is_pushup.param'), ('2e37cd61fd083b0d7b9cd6d11bb37b06d95a1fd6', 'models/action_classification/is_pushup.bin'), ('f267486e12f817da0b810ef03aaad940ab5c5253', 'models/human_pose_detection/Ultralight-Nano-SimplePose.param'), ('505bacafeb0a1607a1ab7096e208106ac74f8ead', 'models/human_pose_detection/Ultralight-Nano-SimplePose.bin'), ('bc26db6250cd857ecdf3b5ad6f62cef05fbb1a7d', 'configs/background_matting_config.json'), ('1b3c28c98e2d9e19112bb604da9a5bbdcea1bba3', 'configs/barcode_scanner_config.json'), ('ddda8dd176666c649e533d36698ca01b4ea3ce72', 'configs/pushup_counter_config.json'), ('ba22b3e291f48ee5247a9871ced71a5187447e54', 'configs/face_detector_config.json')]
}

_split_asset_bins = {}

github_repo_url = "https://github.com/VNOpenAI/daisykit-assets/raw/master/"
_url_format = "{repo_url}{file_name}"


def merge_file(root, files_in, file_out, remove=True):
    with open(file_out, "wb") as fd_out:
        for file_in in files_in:
            file = os.path.join(root, file_in)
            with open(file, "rb") as fd_in:
                fd_out.write(fd_in.read())
            if remove == True:
                os.remove(file)


def short_hash(name):
    if name not in _asset_sha1:
        raise ValueError(
            "Pretrained asset for {name} is not available.".format(name=name)
        )
    return _asset_sha1[name][:8]


def get_asset_file(name, tag=None, root=os.path.join("~", ".daisykit", "assets")):
    r"""Return location for the pretrained on local file system.

    This function will download from online asset zoo when asset cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the asset.
    root : str, default '~/.daisykit/assets'
        Location for keeping the asset parameters.

    Returns
    -------
    file_path
        Path to the requested asset file.
    """
    if "DAISYKIT_HOME" in os.environ:
        root = os.path.join(os.environ["DAISYKIT_HOME"], "assets")

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = "{name}-{short_hash}".format(name=name, short_hash=tag)
    else:
        file_name = "{name}".format(name=name)

    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name)
    lockfile = os.path.join(root, file_name + ".lock")

    # Create folder
    pathlib.Path(lockfile).parents[0].mkdir(parents=True, exist_ok=True)

    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _asset_sha1[name]

    with portalocker.Lock(
        lockfile, timeout=int(os.environ.get(
            "DAISYKIT_ASSET_LOCK_TIMEOUT", 300))
    ):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning(
                    "Hash mismatch in the content of asset file '%s' detected. "
                    "Downloading again.",
                    params_path,
                )
        else:
            logging.info("Asset file not found. Downloading.")

        zip_file_path = os.path.join(root, file_name)
        if file_name in _split_asset_bins:
            file_name_parts = [
                "%s.part%02d" % (file_name, i + 1)
                for i in range(_split_asset_bins[file_name])
            ]
            for file_name_part in file_name_parts:
                file_path = os.path.join(root, file_name_part)
                repo_url = os.environ.get("DAISYKIT_REPO", github_repo_url)
                if repo_url[-1] != "/":
                    repo_url = repo_url + "/"
                download(
                    _url_format.format(repo_url=repo_url,
                                       file_name=file_name_part),
                    path=file_path,
                    overwrite=True,
                )

            merge_file(root, file_name_parts, zip_file_path)
        else:
            repo_url = os.environ.get("DAISYKIT_REPO", github_repo_url)
            if repo_url[-1] != "/":
                repo_url = repo_url + "/"
            download(
                _url_format.format(repo_url=repo_url, file_name=file_name),
                path=zip_file_path,
                overwrite=True,
            )
        if zip_file_path.endswith(".zip"):
            with zipfile.ZipFile(zip_file_path) as zf:
                zf.extractall(root)
            os.remove(zip_file_path)
        # Make sure we write the asset file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError(
                "Downloaded file has different hash. Please try again.")


def purge(root=os.path.join("~", ".daisykit", "assets")):
    r"""Purge all pretrained asset files in local file store.

    Parameters
    ----------
    root : str, default '~/.daisykit/assets'
        Location for keeping the asset parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))
