from django.db import models

# Create your models here.

class User(models.Model):
    id = models.AutoField(primary_key=True)  
    email = models.EmailField(max_length=50) 
    isGuideChecked = models.BooleanField(default=False)
    nickname = models.CharField(max_length=20)
    password = models.CharField(max_length=50, null=True)
    
    class Meta:
        db_table = 'user'