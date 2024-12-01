from django.http import HttpResponse
from .inference import infer

def index(request):
   infer()
   return HttpResponse('Inference is now running in the background.')