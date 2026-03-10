# # """
# # URL configuration for core project.

# # The `urlpatterns` list routes URLs to views. For more information please see:
# #     https://docs.djangoproject.com/en/6.0/topics/http/urls/
# # Examples:
# # Function views
# #     1. Add an import:  from my_app import views
# #     2. Add a URL to urlpatterns:  path('', views.home, name='home')
# # Class-based views
# #     1. Add an import:  from other_app.views import Home
# #     2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
# # Including another URLconf
# #     1. Import the include() function: from django.urls import include, path
# #     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
# # """
# # from django.contrib import admin
# # from django.urls import path

# # urlpatterns = [
# #     path('admin/', admin.site.urls),
# # ]


# from django.contrib import admin
# from django.urls import path
# from communication import views
# from django.conf import settings
# from django.conf.urls.static import static

# urlpatterns = [
#     path('admin/', admin.site.urls),

#     # Home page: Two-way Communication
#     path('', views.home, name='home'),

#     # Speech to Sign page
#     path('speechtosign/', views.speechtosign, name='speechtosign'),

#     # # API endpoint: Convert speech text to ISL gloss
#     # path('convert/', views.convert_speech_to_gloss, name='convert_speech_to_gloss'),
    
#     # API endpoint to convert text to ISL gloss and return JSON  
#     path('convert_speech/', views.convert_speech, name='convert_speech'),
# ]

# # Serve media files during development
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



from django.contrib import admin
from django.urls import path
from communication import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("speechtosign/", views.speechtosign, name="speechtosign"),
    path("signtospeech/", views.signtospeech, name="signtospeech"),
    path("convert_speech/", views.convert_speech, name="convert_speech"),
    path("play-signs/", views.play_signs, name="play_signs"),
    path("api/record/start/", views.record_start, name="record_start"),
    path("api/record/stop/", views.record_stop, name="record_stop"),
    path("api/record/stream/", views.record_stream, name="record_stream"),
    path("api/gloss/sentence/", views.gloss_sentence, name="gloss_sentence"),
    path("api/gloss/code/", views.gloss_code, name="gloss_code"),
    path("api/tts/speak/", views.tts_speak, name="tts_speak"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)