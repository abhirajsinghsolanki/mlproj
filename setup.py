from setuptools import setup, find_packages

def getrequirements(file_path:str)->list[str]:
    '''this function will return the list of requirements'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    if '-e .' in requirements:
        requirements.remove('-e .')
    return requirements
    
setup(
    name='my_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=getrequirements('requirements.txt'),
    author='abhiraj',
    author_email='abhirajsolanki2005@example.com'
        )