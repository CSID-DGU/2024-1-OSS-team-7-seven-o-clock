from django.urls import path
from myapp import views
from web_demo import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.main_view, name='main_view'),
    path('get-datasets', views.get_datasets, name='get_datasets'),
    path('start_re_id', views.start_re_id, name='start_re_id'),
    path('regist_dataset', views.regist_dataset, name='regist_dataset'),
    path('get_state/<str:task_id>', views.get_state, name='get_state'),
    path('getImgs/', views.get_imgs, name='get_imgs'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
