# Script defines function to check valid email pattern and mask the same
import re

regex = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.]+\.[A-Z|a-z]{2,}$'
mask_patten = '(?<=^.)[^@]*|(?<=@.).*(?=\.[^.]+$)'


def check(email_id):
    """
    Check email id against defined regex pattern
    :param email_id:
    :return: 1 or 0
    """
    if re.search(regex, email_id):
        print("Valid Email")
        flag = 1
    else:
        print("Invalid Email")
        flag = 0
    return flag


def check_n_mask(email_id):
    """
    Validate email ID and Mask if Valid
    :param email_id:
    :return:
    """

    valid_flag = check(email_id)
    if valid_flag == 1:
        masked_email = re.sub(mask_patten, "*", email_id)
        print('Masked Email : ' + masked_email)


if __name__ == '__main__':
    email = "hi Xyz.123@ymail.com"
    check_n_mask(email)

    email = "ankita@g-gmail.com"
    check_n_mask(email)

    email = "ankita@gMail.com"
    check_n_mask(email)

    email = "ankita@edu"
    check_n_mask(email)

    email = "ankita@edu.com"
    check_n_mask(email)

    email = "ankita@yahoo.co.in"
    check_n_mask(email)

    email = "@yahoo.co.in"
    check_n_mask(email)
