from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='uploads/')
    converted_text = models.TextField(blank=True)
    converted = models.BooleanField(default=False)
