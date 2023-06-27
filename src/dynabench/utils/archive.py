import numpy as np
import tarfile
import io   # python3 version
import warnings


def write_np_array_to_tar(array, arrayname, tarpath):
    abuf = io.BytesIO()
    np.save(abuf, array)
    abuf.seek(0)

    tar=tarfile.TarFile(tarpath,'a')
    if arrayname in tar.getnames():
        tar.close()
        warnings.warn(f"File {arrayname} already present in tar {tarpath}. Skipping.", RuntimeWarning)
        return
    info= tarfile.TarInfo(name=arrayname)
    info.size=len(abuf.getbuffer())
    tar.addfile(tarinfo=info, fileobj=abuf)
    tar.close()