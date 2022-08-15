
names = ['name1', 'name2', 'name3', 'name4']

for name in names : 
    with open("{}.txt".format(name), "w", encoding="utf8") as email_file:
        email_file.write(f"""
                         들어갈 내용.{name}                         
                         """)
