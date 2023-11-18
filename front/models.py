from django.db import models

class Item(models.model):
    name = models.CharField(max_length = 255)
    description = models.TextField()
