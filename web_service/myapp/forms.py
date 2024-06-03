from django import forms


class StartReIdForm(forms.Form):
    dataset = forms.CharField()
    image = forms.ImageField()

class RegistDataset(forms.Form):
    dataset = forms.CharField()
    video = forms.FileField()