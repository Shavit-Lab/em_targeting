import subprocess
from pathlib import Path

ilastik_path = (
    "/Applications/ilastik-1.4.0-OSX.app/Contents/ilastik-release/run_ilastik.sh"
)


def apply_ilastik(
    ilastik_project_path,
    image_path,
    export_source="Simple Segmentation",
    suffix="ilastik",
):
    image_path = Path(image_path)

    output_file = image_path.parent / f"{image_path.stem}_{suffix}.h5"
    output_file = str(output_file)

    image_path = str(image_path)

    # output_format = f

    command = [
        ilastik_path,
        "--headless",
        f"--project={ilastik_project_path}",
        f"--export_source={export_source}",
        f"--output_filename_format={output_file}",
        image_path,
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_file
