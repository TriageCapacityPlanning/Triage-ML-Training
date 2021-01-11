from setuptools import setup

setup(name='triage_ml',
      version='0.1',
      description='Triage ML model repository and training tool',
      url='https://github.com/TriageCapacityPlanning/Triage-ML-Training',
      packages=['triage_ml'],
      install_requires=['pytest', 'tensorflow-gpu', 'psycopg2', 'numpy'],
      scripts=['bin/train.py'],
      zip_safe=False)
