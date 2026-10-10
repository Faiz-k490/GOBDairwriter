"""Offline SMTP reply classification: no sockets, credentials, or camera."""
import smtplib
import unittest

import mailer


class MailerRetryTests(unittest.TestCase):
    def deliver(self, first_error):
        attempts = []

        def transport(config, message):
            attempts.append(message)
            if len(attempts) == 1:
                raise first_error

        # Run the real delivery loop synchronously with the injected transport.
        sender = mailer.EmailSender(
            dict(user="sender@example.test", enabled=False),
            transport=transport, backoff=(0,),
        )
        sender._deliver(dict(id="record", to="recipient@example.test",
                             png="", ascii="fixture", when=""))
        return sender.results(), len(attempts)

    def test_temporary_authentication_failure_retries(self):
        result, attempts = self.deliver(smtplib.SMTPAuthenticationError(454, b"try later"))
        self.assertEqual(result, [("record", "sent", "", 2)])
        self.assertEqual(attempts, 2)

    def test_temporary_sender_refusal_retries(self):
        error = smtplib.SMTPSenderRefused(451, b"try later", "sender@example.test")
        result, attempts = self.deliver(error)
        self.assertEqual(result, [("record", "sent", "", 2)])
        self.assertEqual(attempts, 2)

    def test_temporary_recipient_refusal_retries(self):
        error = smtplib.SMTPRecipientsRefused({"recipient@example.test": (450, b"try later")})
        result, attempts = self.deliver(error)
        self.assertEqual(result, [("record", "sent", "", 2)])
        self.assertEqual(attempts, 2)

    def test_permanent_failures_stop_after_one_attempt(self):
        errors = [
            smtplib.SMTPAuthenticationError(535, b"rejected"),
            smtplib.SMTPSenderRefused(550, b"rejected", "sender@example.test"),
            smtplib.SMTPRecipientsRefused({"recipient@example.test": (550, b"rejected")}),
            smtplib.SMTPDataError(554, b"rejected"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                result, attempts = self.deliver(error)
                self.assertEqual(result[0][1], "failed")
                self.assertEqual(result[0][3], 1)
                self.assertEqual(attempts, 1)

    def test_transport_timeout_still_retries(self):
        result, attempts = self.deliver(TimeoutError("offline fixture"))
        self.assertEqual(result, [("record", "sent", "", 2)])
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
