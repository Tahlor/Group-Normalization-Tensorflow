""" Utilities """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Logging
# =======

import logging
from colorlog import ColoredFormatter
import tensorflow as tf


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('Log')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov


def train_test_summary(name, value, max_outputs=4, summary_type='scalar'):
    if summary_type == 'scalar':
        tf.summary.scalar(name, value, collections=['train'])
        tf.summary.scalar("test_{}".format(name), value, collections=['test'])
    elif summary_type == 'image':
        tf.summary.image(name, value, max_outputs=max_outputs, collections=['train'])
        tf.summary.image("test_{}".format(name), value,
                         max_outputs=max_outputs, collections=['test'])


def email(to_email, from_email, subj, msg_text="", html=True, useBWserver=False):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    if html:
        msg_text = msg_text.replace('\n', '<br>')
        msg_text = msg_text.replace('\t', '&nbsp;' * 5)
    # use <pre> and </pre>
    msg = MIMEMultipart("alternative")
    # msg.attach(MIMEText(msg_text, 'plain', 'utf-8'))
    msg.attach(MIMEText(msg_text, 'html', 'utf-8'))
    msg['Subject'] = subj
    msg['From'] = from_email
    msg['To'] = to_email

    if not useBWserver:
        username = 'bateswhitemailer'
        password = 'bwguest1234'
        s = smtplib.SMTP('smtp.gmail.com:587')
        s.ehlo()
        s.starttls()
        s.login(username, password)
        from_email = "bateswhitemailer@gmail.com"
    s.sendmail(from_email, [to_email], msg.as_string())
    print("Done with email")
    s.quit()