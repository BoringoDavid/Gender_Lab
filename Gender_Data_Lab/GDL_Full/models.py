from django.db import models
# Create your models here.
from django.db import models
from django.core.exceptions import ValidationError
from django.utils import timezone
import os

class DataFileUpload(models.Model):
    FILE_TYPES = [
        ('csv', 'CSV'),
        ('xls', 'Excel (.xls)'),
        ('xlsx', 'Excel (.xlsx)'),
        ('dta', 'Stata (.dta)'),
        ('sav', 'SPSS (.sav)'),
        ('sps', 'SPSS (.sps)'),
        ('spv', 'SPSS (.spv)'),
    ]

    SOURCE_TYPES = [
        ('internal', 'Internal'),
        ('external', 'External'),
    ]

    TIME_FREQUENCY_CHOICES = [
        ('After 5 years', 'After 5 years'),
        ('After 4 years', 'After 4 years'),
        ('After 3 years', 'After 3 years'),
        ('After 2 years', 'After 2 years'),
        ('annual', 'Annual'),
        ('quarterly', 'Quarterly'),
        ('monthly', 'Monthly'),
    ]

    file = models.FileField(upload_to='data_uploads/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPES, editable=False)
    source_type = models.CharField(max_length=10, choices=SOURCE_TYPES, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Internal data source fields
    survey_name = models.CharField(max_length=255, blank=True, null=True)
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    time_frequency = models.CharField(max_length=20, choices=TIME_FREQUENCY_CHOICES, blank=True, null=True)
    internal_description = models.TextField(
        blank=True,
        null=True,
        help_text="Indicators defined by admin (predefined before Elasticsearch)"
    )

    # External data source fields
    external_file_name = models.CharField(max_length=255, blank=True, null=True)
    file_section_from = models.CharField(max_length=255, blank=True, null=True)
    external_description = models.TextField(blank=True, null=True)

    def clean(self):
        today = timezone.now().date()
        if self.start_date and self.start_date > today:
            raise ValidationError("Start date cannot be in the future.")
        if self.end_date and self.end_date > today:
            raise ValidationError("End date cannot be in the future.")

        # Validate and assign file_type
        if self.file:
            ext = os.path.splitext(self.file.name)[1][1:].lower()
            if ext not in dict(self.FILE_TYPES):
                raise ValidationError(f"Unsupported file type: .{ext}")
            self.file_type = ext

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.file.name} ({self.source_type})"



# SURVEY TABLE 

# Validator to prevent future years
def validate_year_only(value):
    if value > timezone.now().date():
        raise ValidationError("Future years are not allowed.")

# === 1. Survey Model ===
class Survey(models.Model):
    survey_id = models.AutoField(primary_key=True)
    
    FILE_TYPES = [
        ('csv', 'CSV'),
        ('xls', 'Excel (.xls)'),
        ('xlsx', 'Excel (.xlsx)'),
        ('dta', 'Stata (.dta)'),
        ('sav', 'SPSS (.sav)'),
        ('sps', 'SPSS (.sps)'),
        ('spv', 'SPSS (.spv)'),
    ],
    SURVEY_NAME_CHOICES = [
        ('integrated_business_survey', 'Integrated Business Enterprise Survey'),
        ('eicv', 'Integrated Household Living Conditions Survey (EICV)'),
        ('labour_force_survey', 'Labour Force Survey'),
    ]

    file = models.FileField(upload_to='survey_files/')
    survey_name = models.CharField(max_length=100, choices=SURVEY_NAME_CHOICES)
    year = models.DateField(validators=[validate_year_only])

    def __str__(self):
        return f"{self.survey_name} ({self.year.year})"

# === 2. Indicator Model ===
class Indicator(models.Model):
    survey = models.ForeignKey(Survey, on_delete=models.CASCADE, related_name="indicators")
    indicator_name = models.CharField(max_length=250)

    def __str__(self):
        return self.indicator_name

        