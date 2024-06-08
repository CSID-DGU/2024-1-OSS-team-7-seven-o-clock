from django import forms


class StartReIdForm(forms.Form):
    dataset_name = forms.CharField()
    query_file = forms.ImageField()

class RegistDataset(forms.Form):
    dataset_name = forms.CharField()
    dataset_base = forms.FileField()