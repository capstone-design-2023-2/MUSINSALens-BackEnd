from django.db import models

class UploadedImage(models.Model):
    id = models.AutoField(primary_key=True)
    image_path = models.CharField(max_length=1000)
    file_name = models.CharField(max_length=255)
    
    class Meta:
        db_table = 'upload_image'

class Product(models.Model):
    id = models.AutoField(primary_key=True)
    image_url = models.CharField(max_length=255)
    info_url = models.CharField(max_length=255)
    brand = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    # price = models.DecimalField(max_digits=10, decimal_places=2)
    price = models.CharField(max_length=255)
    # original_price = models.DecimalField(max_digits=10, decimal_places=2)
    original_price = models.CharField(max_length=255)
    category = models.CharField(max_length=255)
    sub_category = models.CharField(max_length=255)
    code = models.CharField(max_length=255)
    image_path = models.ImageField(upload_to='images/')

    class Meta:
        db_table = 'product'