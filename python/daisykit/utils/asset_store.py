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
    for checksum, name in [('069623b785bc2a0850bce39f0f5ef48a2c37f758', 'images/background.jpg'), ('a502c0739d1e415bd10e92132565b32b0e9d89d0', 'models/face_antispoofing/minivision/README.md'), ('4454a5d20dc288e8d40bdf9f1170fba430af672c', 'models/face_antispoofing/minivision/model_2.bin'), ('aba368bab943d491a583e100cb46680707cc7d85', 'models/face_antispoofing/minivision/model_2.param'), ('123796493edd28c04b64630a264d5a3d1117a8e9', 'models/face_extraction/arcface/iresnet18_1.bin'), ('790de300d1899120e5fe8932da7ae5383af0f45a', 'models/face_extraction/arcface/README.md'), ('a7abe0c6869270300525a4ad3bf2cb5324ec8b68', 'models/face_extraction/arcface/iresnet18_1.param'), ('3e57a3fbb29f530411c5fe566c5d8215b05b1913', 'models/facial_landmark/pfld/pfld-sim.param'), ('4d44b100fa1cfcbff97a0c5785738912556d441a', 'models/facial_landmark/pfld/README.md'), ('cf8cc7a623ba31cb8c700d59bf7bd622f081d695', 'models/facial_landmark/pfld/pfld-sim.bin'), ('272fce902d7851432ebf5936771726415302298f', 'models/hand_detection/yolox/README.md'), ('24455837664448b4ad00da95d797c475303932f7', 'models/hand_detection/yolox/yolox_hand_swish.bin'), ('5c2d4a808bc5a99c5bde698b1e52052ec0d0eee4', 'models/hand_detection/yolox/yolox_hand_relu.param'), ('c9fe82e4c0fe6ee274757cece1a9269578806282', 'models/hand_detection/yolox/yolox_hand_relu.bin'), ('7fa457c0a355fd56b9fd72806b8ba2738e1b1dda', 'models/hand_detection/yolox/yolox_hand_swish.param'), ('77afc7d409bf9c23863f092c230b8dfd7b8059d0', 'models/background_matting/human_segmentation_pphumanseg_2023mar/human_segmentation_pphumanseg_2023mar-sim-opt.bin'), ('2fde32e728009d9386f8ce99fa0e189232d2d940', 'models/background_matting/human_segmentation_pphumanseg_2023mar/human_segmentation_pphumanseg_2023mar-sim-opt.param'), ('1a348a9aec890c6331c5d4a4f45c321c2a304365', 'models/background_matting/human_segmentation_pphumanseg_2023mar/README.md'), ('adee101923317578dc8fa41ca680fd9c6f877187', 'models/background_matting/erd/erdnet.param'), ('3fccfbd53ed2907fc51735313dcabb7b323451ad', 'models/background_matting/erd/README.md'), ('c54b251a44fb0e542ff77ea3bac79883a63bc8d7', 'models/background_matting/erd/erdnet.bin'), ('dd3a085ceaf7efe96541772a1c9ea28b288ded0c', 'models/object_detection/yolox/yolox-tiny.bin'), ('aa4f38ccdd601585ae6e6a2af14305b1d8dde078', 'models/object_detection/yolox/README.md'), ('aef43afeef44c045a4173a45d512ff98facfdfcb', 'models/object_detection/yolox/yolox-nano.bin'), ('be69b7a08c2decc3de52cfc7f7086bc8bc4046f3', 'models/object_detection/yolox/yolox-tiny.param'), ('836cb7b908db231c4e575c2ece9d90f031474a37', 'models/object_detection/yolox/yolox-nano.param'), ('1ab6d93570e408e8c85c759ee1c61d67cb9509b6', 'models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.bin'), ('7451cb1978b7f51a2cc92248e9c103fd456e0f74', 'models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.param'), ('ffaa5614bbadcb9980641b67da1df551fa4783d4', 'models/face_detection/yolo_fastest_with_mask/README.md'), ('a08ff981deac55acbff84aad8bb5fcb6ba952631', 'models/face_detection/scrfd/scrfd_2.5g_1.param'), ('530128d59ad44260b217031e21427aaa1d9b076b', 'models/face_detection/scrfd/README.md'), ('ced0b236094a8c8d7328ae2a168f382d6388096a', 'models/face_detection/scrfd/scrfd_2.5g_1.bin'), ('8d0ee0cfe95843d72c7da1948ae535899ebd2711', 'models/hand_pose/mediapipe_hand/hand_lite-op.param'), ('f09f29e615a92bf8f51a6ed35b6fc91cb4778c54', 'models/hand_pose/mediapipe_hand/hand_lite-op.bin'), ('952c782460cd2b98ab7a11ea792693d9c3fa1458', 'models/hand_pose/mediapipe_hand/README.md'), ('e859e4766d5892085678c253b06324e463ab4728', 'models/hand_pose/mediapipe_hand/hand_full-op.bin'), ('cf4497bf1ebd69f9fe404fbad765af29368fbb45', 'models/hand_pose/mediapipe_hand/hand_full-op.param'), ('88d4e03b3cccc25de4f214ef55d42b52173653c0', 'models/human_detection/ssd_mobilenetv2/ssd_mobilenetv2.bin'), ('94454595462d86978f04e10ee9e0bcc0e04ce390', 'models/human_detection/ssd_mobilenetv2/README.md'), ('c4a952480ea26c1e2097e80f5589df707f31463c', 'models/human_detection/ssd_mobilenetv2/ssd_mobilenetv2.param'), ('d6b50b69a7e7c9ae8614f34f643c006b35daf3fd', 'models/human_pose_detection/movenet/lightning.bin'), ('78bd88d738dcc7edcc7bc570e1456ae26f9611ec', 'models/human_pose_detection/movenet/thunder.param'), ('1af4dcb8b69f4794df8b281e8c5367ed0ee38290', 'models/human_pose_detection/movenet/thunder.bin'), ('906d22b1a47f082907621bc708f9ed51ffea5de4', 'models/human_pose_detection/movenet/README.md'), ('dc944ed31eaad52d6ef0bf98852aa174ed988796', 'models/human_pose_detection/movenet/lightning.param'), ('a11625824e7dbda58abcf88cb65f24d8858d022b', 'README.md'), ('39bf6421f398f0a1e5418738705f5b93e3701cf1', 'configs/hand_pose_yolox_mp_config.json'), ('b681cb0081900d3be63a9cd84bcb7195123f8078', 'configs/background_matting_config.json'), ('1b3c28c98e2d9e19112bb604da9a5bbdcea1bba3', 'configs/barcode_scanner_config.json'), ('102e2cd0ef2336a9c794a56a82a757c2049c3ac7', 'configs/object_detector_yolox_config.json'), ('91477d8ca033d8321e7ae241150c2ef278689ce1', 'configs/human_pose_movenet_config.json'), ('724e7330422c691a47917ee7effacac07db035a2', 'configs/face_detector_config.json')]
}

_split_asset_bins = {}

github_repo_url = "https://github.com/nrl-ai/daisykit-assets/raw/master/"
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
