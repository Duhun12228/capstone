import rospy
import math
from ackermann_msgs.msg import AckermannDriveStamped
import time

class HardCode:

    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                                AckermannDriveStamped, queue_size=1)
        rospy.init_node('turn_test_node', anonymous=True)
        # ---------- PARAMETERS ----------
        # Speeds
        self.forward_speed = rospy.get_param("~forward_speed", 0.22)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.22)

        # Sharp turn steering angle
        self.sharp_left = rospy.get_param("~sharp_left", math.radians(43.0))
        self.straight = 0.0

        # Timings (tune on track)
        self.turn_into_time = rospy.get_param("~turn_into_time", 2.5)       # swing left
        self.return_time = rospy.get_param("~return_time",2.7)             # return back
        self.forward_105_time = rospy.get_param("~forward_105_time", 4.5)   # forward 1.05m

        # publish rate
        self.control_rate = rospy.get_param("~control_rate", 20.0)

    # --- Low-level send ---
    def send_cmd(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.cmd_vel_pub.publish(msg)

    # --- Hold movement for duration ---
    def timed_move(self, speed, steering, duration):
        rate = rospy.Rate(self.control_rate)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.send_cmd(speed, steering)
            rate.sleep()

        # brief brake
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.15)

    # ============== NEW 5-STEP ALGORITHM ==============
    def turn_left(self):
        rospy.loginfo("=== NEW OFFICIAL PARKING START ===")
        rospy.sleep(3.0)
        # --------------------------
        # STEP 1 → Sharp left turn into the parking area
        # --------------------------
        rospy.loginfo("STEP 1: Sharp LEFT swing into parking zone")
        self.timed_move(self.forward_speed,0,2.0)
        self.timed_move(self.forward_speed, self.sharp_left, self.turn_into_time)

        # --------------------------
        # STEP 2 → STOP & stay inside for 3 seconds
        # --------------------------
        rospy.loginfo("STEP 2: STOPPING inside parking zone for REQUIRED 3 seconds")
        self.send_cmd(0.0, 0.0)   # full stop
        rospy.sleep(0.5)

        # --------------------------
        # STEP 3 → Return back to white line (reverse)
        # --------------------------
        rospy.loginfo("STEP 3: Returning back to white line")
        self.timed_move(self.forward_speed,0.0,1.0)

        # --------------------------
        # STEP 5 → Final stop / mission complete
        # --------------------------
        rospy.loginfo("=== PARKING MISSION COMPLETE ===")

        self.send_cmd(0.0, 0.0)
        rospy.sleep(3.0)

    def turn_right(self):
        pass


if __name__ == '__main__':
    try:
        lf = HardCode()
        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            lf.turn_left()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass