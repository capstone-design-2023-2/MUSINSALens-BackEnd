from django.db import models

class UploadedImage(models.Model):
    id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=255)
    image_path = models.ImageField(upload_to='uploaded_images/')

    class Meta:
        db_table = 'upload_image'