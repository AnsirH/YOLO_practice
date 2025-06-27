from roboflow import Roboflow
rf = Roboflow(api_key="2VWxqz3VGiGTnAFVywmy")
project = rf.workspace("defect-detecting").project("my-first-project-pqbdl")
version = project.version(3)
dataset = version.download("yolov11")