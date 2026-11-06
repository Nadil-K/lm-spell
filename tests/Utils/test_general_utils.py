from lmspell.utils.general_utils import GeneralUtils


def test_delete_folder(isolated_cwd):
    folder = isolated_cwd / "exp" / "epoch_1"
    folder.mkdir(parents=True)

    GeneralUtils.delete_folder(str(isolated_cwd / "exp"))
    GeneralUtils.delete_folder(str(isolated_cwd / "exp"))  # deleting a missing folder is a no-op

    assert not (isolated_cwd / "exp").exists()
