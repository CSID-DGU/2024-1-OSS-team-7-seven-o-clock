from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.core.files.uploadedfile import TemporaryUploadedFile as BaseTemporaryUploadedFile

class TemporaryUploadedFile(BaseTemporaryUploadedFile):
    def close(self) -> None:
        print('close')
        return