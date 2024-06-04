from django.core.files.uploadhandler import TemporaryFileUploadHandler as BaseTemporaryFileUploadHandler

from myapp.uploadedfile import TemporaryUploadedFile

class TemporaryFileUploadHandler(BaseTemporaryFileUploadHandler):
    def new_file(self, *args, **kwargs):
        super().new_file(*args, **kwargs)
        self.file = TemporaryUploadedFile(
            self.file_name, self.content_type, 0, self.charset, self.content_type_extra
        )