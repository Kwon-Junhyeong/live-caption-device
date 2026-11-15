from pathlib import Path

from setuptools import setup

package = "stt_bridge_node"


def _vendor_files(root: str):
    entries = []
    base = Path(root)
    for path in sorted(base.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        entries.append((f"share/{package}/{path.parent.as_posix()}", [path.as_posix()]))
    return entries


setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", [str(p) for p in Path("launch").glob("*.py")]),
    ] + _vendor_files("stt_module"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Junhyeong Kwon",
    maintainer_email="meoruu00@gmail.com",
    description="Runs one streaming speech recognition session per speaker track and publishes captions",
    license="MIT",
    entry_points={"console_scripts": [f"stt_bridge = {package}.node:main"]},
)
