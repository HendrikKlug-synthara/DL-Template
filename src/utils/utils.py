def get_callback(callback_list, which_cb):
    return [cb for cb in callback_list if which_cb in str(cb)][0]
