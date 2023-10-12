from django.conf import settings
from django.conf.urls.static import static

from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.index, name='index'),
    path('download/', views.download_df, name='download_df'),
    path('test/', views.test),
    path('chart/', views.chart, name='base'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
