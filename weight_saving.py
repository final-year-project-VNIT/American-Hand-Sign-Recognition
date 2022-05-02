model_json = classifier.to_json()
with open('/content/drive/MyDrive/Models/model-bw.json', "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('/content/drive/MyDrive/Models/model-bw.h5')
print('Weights saved')