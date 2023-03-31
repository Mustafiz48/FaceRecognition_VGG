# Face Recognition with VGGFace

This is is a face recognition project with VGGFace
Here I have used VGG architecture to recognize people. 

### Install required files
Install the necesasry packages with the following command.

```bash
  pip install -r requirements.txt
```

### Face Registration 
First we need to register peoples. To do that, go to "Registration_Images" folder and create folder for each person. Inside the folders, keep a picture of that person with a clear face. Run the following command from project directory

```bash
  python main.py "registration"
```
The command will perform Registration. A success message will be printed at the end of successfull registration. 

### Face Recognition
After the registration is complete, you can perform recognitionl. To do that, run the following command
```bash
  python main.py "recognition"
```

After you run the command, I will be asked to give path to the image you want to perform recognition. Give the image path, it will show you the result.

N.B Recognition of "unknown" person is not implemented yet. It can easily be done by comparing the minimmum distance (self.distance variable in Facerecognizer class) with a threshold value. Need to find the threshold value for good performance
