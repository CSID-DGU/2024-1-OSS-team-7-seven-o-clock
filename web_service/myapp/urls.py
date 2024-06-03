from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.main_view, name='main_view'),
    path('start_re_id', views.start_re_id, name='start_re_id'),
    path('regist_dataset', views.regist_dataset, name='regist_dataset'),
    path('get_state/<str:task_id>/', views.get_state, name='get_state'),
    # path('add_task_example', views.add_task_example, name='add_task_example'),
]
