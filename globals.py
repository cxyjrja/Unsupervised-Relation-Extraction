#Defines a triple relationship
#The relationship has a text component, for example "subsidiary of"
#And an attribute component. Attributes are stored in the form of a dictionary
#Example: "time": November 8, 2018

class TripleRelationship():
    def __init__(self,rel):
        self.relation = rel
        self.attributes = dict()

    def set_attribute(self,attribute_name,value):
        self.attributes[attribute_name] = value

    def __repr__(self):
        return self.relation

    def __str__(self):
        return self.relation
    
    def __eq__(self,other):
        return self.relation == other.relation and self.attributes == other.attributes
    
    def __ne__(self,other):
        return not self.relation == other.relation or not self.attributes == other.attributes

class Entity:
    def __init__(self,entity):
        self.entity = entity
        self.attributes = dict()
        
    def set_attribute(self,attribute_name,value):
        self.attributes[attribute_name] = value

    def __repr__(self):
        return self.entity

    def __str__(self):
        return self.entity
    
    def __eq__(self,other):
        return self.entity == other.entity and self.attributes == other.attributes
    
    def __ne__(self,other):
        return not self.entity == other.entity or not self.attributes == other.attributes

#Given a directory path, returns a list of txt files in that directory
def get_story_list(path):
    """
    @Param:
    @path: file directory ('txt_files_cleaned/*.txt')
    
    @Return: a list of docs in that directory
    """
    
    import glob
    file_lst = glob.glob(path+'/*.txt')
    story_lst = list()

    for file in file_lst:
        with open(file, 'r') as myfile:
            data = myfile.read().replace('\n', '')
            filename = file.split('\\')[-1]
            story_lst.append((filename,data))
    
    return story_lst

#Check whether a triple subject or object contains a named entity. Return the ne data and the
#entity descriptor
def verify_triple(triple,ne_data,left=False,right=False):
    left_val = tuple()
    right_val = tuple()
    if left:
        for n in ne_data:
            if ne_data[n]['name'] in triple[0]:
                left_val = (n,ne_data[n])
                break
        else:
            return False
    if right:
        for n in ne_data:
            if ne_data[n]['name'] in triple[2]:
                right_val =  (n,ne_data[n])
                break
        else:
            return False
    return(left_val,right_val)


#Find an entity from the named_entities
def find_entity(ne_data,text):
    for n in ne_data:
        if ne_data[n]['name'] in text:
            return n,ne_data[n]
    return ''