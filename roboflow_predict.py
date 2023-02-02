from roboflow import Roboflow

rf = Roboflow(api_key="ic9DOyzgD6um05sLint1")
project = rf.workspace().project("macau-vehicle-cv-segmen")
model = project.version(2).model

# infer on local image
print(model.predict("./european-license-plate.jpg").json())
model.predict("./european-license-plate.jpg").save("prediction.jpg")
